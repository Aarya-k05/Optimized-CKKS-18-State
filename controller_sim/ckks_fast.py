"""
============================================================
FAST CKKS BUILDING BLOCKS (TenSEAL / Microsoft SEAL backend)
============================================================
Profiling the original controller step (18 states, N=8192) showed:

    encrypt  ~ 5 ms   -> almost all of it is encryption randomness
                         (public-key encrypt-of-zero), independent of x(k)
    dot      ~ 10-14ms-> ~85 % is the rotate-and-add inside sum()
                         (key-switching), the multiply itself is ~1.5 ms
    decrypt  ~ 1 ms

What is (and is NOT) parallelisable
-----------------------------------
* Across control steps: NO.  x(k+1) = A x(k) + B u(k) needs u(k), which needs
  the decryption of the encryption of x(k).  The loop is a strict data
  dependency, so step k+1 cannot start encrypting before step k finishes.
* The 18 states are SIMD-packed into ONE ciphertext, so there is no
  "18 encryptions" to run in parallel.
* What IS independent of x(k): the randomness-heavy part of encryption.
  CKKS encryption is linear in the message:

        Enc(x) = Enc(0) + Encode(x)

  Enc(0) can be generated ahead of time, by worker threads, while the plant
  is idle (the control period is 100 ms, the step needs ~10-20 ms).  The
  online encryption then costs only encode + one polynomial addition.
  Every Enc(0) is used exactly once, so this is standard CKKS encryption,
  not a security shortcut.

Modes (see EncryptedController)
-------------------------------
  baseline : ts.ckks_vector(ctx, x) ; enc.dot(-K)         (original code path)
  precomp  : Enc(0) pool + add_plain ; enc.dot(-K)        (still full HE dot)
  norot    : Enc(0) pool + add_plain ; enc * (-K) ; the key holder sums the
             18 slots after decryption.  No rotations / Galois keys at all.
             TRADE-OFF: the key holder sees the 18 products K_i*x_i, so it can
             learn K.  Use only if the gain K is not secret.
============================================================
"""

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tenseal as ts

# CKKS parameter profiles (all N = 8192 => 128-bit security for <= 218 bits)
PROFILES = {
    # original parameters from the project
    "paper": dict(poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60],
                  global_scale=2 ** 40),
    # one fewer 40-bit prime: the dot product only consumes ONE level
    # (mul_plain + rescale), so the 2nd 40-bit prime is dead weight.  Same
    # scale / same precision, ~25 % fewer RNS limbs in every NTT, rotation
    # and ciphertext.
    "lean": dict(poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 60],
                 global_scale=2 ** 40),
}


def make_context(profile="paper", galois=True):
    p = PROFILES[profile]
    ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                     poly_modulus_degree=p["poly_modulus_degree"],
                     coeff_mod_bit_sizes=list(p["coeff_mod_bit_sizes"]))
    ctx.global_scale = p["global_scale"]
    if galois:
        ctx.generate_galois_keys()   # only needed for the rotate-and-sum
    return ctx


class EncZeroPool:
    """
    Pool of fresh encryptions of zero.

    fill()   : generate up to `target` entries (optionally on worker threads)
    pop()    : take one (falls back to generating one inline = a "miss")
    """

    def __init__(self, ctx, n_slots, target=4, workers=1, threaded=False):
        self.ctx, self.n, self.target = ctx, n_slots, target
        self._q = deque()
        self._lock = threading.Lock()
        self.misses = 0
        self.offline_ms = []          # cost of each Enc(0) (off critical path)
        self._ex = ThreadPoolExecutor(max_workers=workers) if threaded else None
        self._pending = 0

    def _make(self):
        t = time.perf_counter()
        z = ts.ckks_vector(self.ctx, [0.0] * self.n)
        dt = (time.perf_counter() - t) * 1e3
        with self._lock:
            self._q.append(z)
            self.offline_ms.append(dt)
            self._pending = max(0, self._pending - 1)

    def fill(self):
        """Top the pool up (call this in the idle part of the control period)."""
        with self._lock:
            need = self.target - len(self._q) - self._pending
            self._pending += max(0, need)
        for _ in range(max(0, need)):
            if self._ex is None:
                self._make()
            else:
                self._ex.submit(self._make)

    def pop(self):
        with self._lock:
            if self._q:
                return self._q.popleft()
        self.misses += 1
        t = time.perf_counter()
        z = ts.ckks_vector(self.ctx, [0.0] * self.n)
        self.offline_ms.append((time.perf_counter() - t) * 1e3)
        return z

    def close(self):
        if self._ex is not None:
            self._ex.shutdown(wait=True)


class EncryptedController:
    """
    One control step = sensor-side encrypt  ->  controller-side HE eval
                       ->  actuator-side decrypt.
    Each stage is timed separately and only the critical path is timed.
    """

    MODES = ("baseline", "precomp", "norot")

    def __init__(self, ctx, K, mode="baseline", pool_target=4, pool_workers=1,
                 pool_threaded=False):
        assert mode in self.MODES
        self.ctx, self.mode = ctx, mode
        self.neg_K = (-np.asarray(K)).flatten().tolist()
        self.n = len(self.neg_K)
        self.pool = None
        if mode in ("precomp", "norot"):
            self.pool = EncZeroPool(ctx, self.n, pool_target, pool_workers,
                                    pool_threaded)
            self.pool.fill()

    # ---- sensor side -------------------------------------------------
    def encrypt(self, x):
        xs = np.asarray(x, dtype=float).flatten().tolist()
        if self.mode == "baseline":
            return ts.ckks_vector(self.ctx, xs)
        return self.pool.pop() + xs          # Enc(0) + Encode(x)

    # ---- controller side (sees ciphertexts only) ---------------------
    def evaluate(self, enc_x):
        if self.mode == "norot":
            return enc_x * self.neg_K        # slot-wise K_i * x_i, no rotation
        return enc_x.dot(self.neg_K)         # mul_plain + rotate-and-sum

    # ---- actuator side -----------------------------------------------
    def decrypt(self, enc_u):
        vals = enc_u.decrypt()
        if self.mode == "norot":
            return float(sum(vals))          # slot sum done by the key holder
        return float(vals[0])

    def idle(self):
        """Call once per control period AFTER the actuator has been updated."""
        if self.pool is not None:
            self.pool.fill()

    def close(self):
        if self.pool is not None:
            self.pool.close()
