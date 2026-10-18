"""
============================================================
ENCRYPTED CLOSED-LOOP SIMULATION
CKKS homomorphic state-feedback controller
============================================================

This file is the actual execution/demo of the encrypted controller.

At every control step:

    1. The simulated plant has a real state x(k).
    2. The sensor side encrypts x(k) using CKKS.
    3. The controller evaluates Enc(x(k)) . (-K).
       The controller-side operation is performed on ciphertext.
    4. Only the resulting control signal is decrypted.
    5. The plaintext plant is updated using that control signal.
    6. The process repeats.

The terminal output deliberately shows the execution pipeline so that
the simulation is observable instead of being only a final graph.

Important:
    This is a software proof-of-concept. The plant itself is simulated
    in plaintext. The state-feedback calculation is the part evaluated
    homomorphically. A real deployment would separate the sensor/
    encryption side, encrypted controller, and decryption/actuator side.

What changed for the 2-state -> 18-state upgrade:
    Nothing in the HE pipeline itself. `enc_x.dot(neg_K)` already
    performs a rotate-and-sum reduction internally, so it works the
    same way whether x has 2 slots or 18. The only change in this
    file is that the per-step trace now stores the FULL state vector
    (state_before / state_after) instead of two hardcoded fields
    named "position"/"velocity", so logging still makes sense for an
    arbitrary number of states.
============================================================
"""

import os
import time
import numpy as np
import tenseal as ts

from state_space_model import build_plant, design_state_feedback, discretize, N_MASSES
from ckks_fast import EncryptedController


def make_context(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 60),
                 galois=True):
    """
    Create the CKKS context used by the simulation.

    Default coefficient modulus is (60, 40, 60): the dot product only consumes
    one level, so the original second 40-bit prime was unused. Same N, same
    scale (2^40), same 128-bit security, same accuracy, ~2x faster HE.
    Pass coeff_mod_bit_sizes=(60, 40, 40, 60) to reproduce the original context.
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=list(coeff_mod_bit_sizes),
    )
    ctx.global_scale = 2 ** 40
    if galois:                      # rotation keys: needed for dot() (not for mode="norot")
        ctx.generate_galois_keys()
    return ctx


def _fmt_vector(values, digits=6):
    values = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{v:.{digits}f}" for v in values) + "]"


def simulate_encrypted(
    Ad,
    Bd,
    K,
    x0,
    steps,
    ctx=None,
    verbose=True,
    demo_steps=8,
    print_every=10,
    mode="precomp",
    pool_threaded=False,
    pool_workers=2,
):
    """
    Run the CKKS encrypted-controller simulation.

    mode:
        "baseline" : exactly the original path (ts.ckks_vector + dot)
        "precomp"  : (default) Enc(x) = pre-generated Enc(0) + Encode(x); the
                     dot product is still fully homomorphic, K stays private
        "norot"    : no rotations; the key holder sums the 18 slots after
                     decryption (fastest, but it can then recover K)

    The optional terminal demonstration prints the first `demo_steps`
    control cycles and then every `print_every` steps. The complete
    numerical trajectory is returned to the caller.

    Returns:
        X         : plant state trajectory
        U         : control signal trajectory
        timings   : encryption/HE/decryption timing lists
        trace     : per-step execution metadata for CSV/reporting
    """
    if ctx is None:
        ctx = make_context()

    ctrl = EncryptedController(ctx, K, mode=mode, pool_threaded=pool_threaded,
                               pool_workers=pool_workers)
    n = Ad.shape[0]
    x = np.array(x0, dtype=float).reshape(n, 1)

    X = np.zeros((steps + 1, n))
    U = np.zeros(steps)
    X[0] = x.flatten()

    timings = {"encrypt": [], "he_dot": [], "decrypt": [], "offline": []}
    trace = []

    if verbose:
        print("\n" + "=" * 72)
        print("CKKS ENCRYPTED STATE-FEEDBACK EXECUTION")
        print("=" * 72)
        print(f"State dimension: {n}   Controller operation: Enc(u) = Enc(x) . (-K)")
        print(f"CKKS execution mode: {mode}")
        print("The plaintext state is NOT used in the controller operation.")
        print("The terminal is an experiment observer; the simulated")
        print("sensor/plant values are shown here for demonstration.")
        print("-" * 72)

    for k in range(steps):
        time_s = k * 0.1

        # ------------------------------------------------------
        # SENSOR -> ENCRYPTION
        # ------------------------------------------------------
        x_before = x.flatten().copy()

        t0 = time.perf_counter()
        enc_x = ctrl.encrypt(x_before)           # sensor side: Enc(x(k))
        t1 = time.perf_counter()

        # ------------------------------------------------------
        # ENCRYPTED CONTROLLER
        # ------------------------------------------------------
        t2 = time.perf_counter()
        enc_u = ctrl.evaluate(enc_x)             # Enc(x) . (-K)
        t3 = time.perf_counter()

        # ------------------------------------------------------
        # DECRYPT ONLY THE CONTROL ACTION
        # ------------------------------------------------------
        u_val = ctrl.decrypt(enc_u)
        t4 = time.perf_counter()

        # Serialized sizes (evidence a ciphertext exists). Measured AFTER all
        # timers stop, so serialisation no longer inflates encrypt/decrypt time.
        try:
            enc_x_bytes = len(enc_x.serialize())
            enc_u_bytes = len(enc_u.serialize())
        except Exception:
            enc_x_bytes = enc_u_bytes = -1

        encrypt_ms = (t1 - t0) * 1e3
        he_ms = (t3 - t2) * 1e3
        decrypt_ms = (t4 - t3) * 1e3

        timings["encrypt"].append(encrypt_ms)
        timings["he_dot"].append(he_ms)
        timings["decrypt"].append(decrypt_ms)

        # Plaintext value is calculated ONLY for validation/measurement.
        # It is not used to produce u_val.
        u_plain_check = float((-K @ x)[0, 0])
        control_error = abs(u_plain_check - u_val)

        # ------------------------------------------------------
        # PHYSICAL PLANT UPDATE
        # ------------------------------------------------------
        x = Ad @ x + Bd * u_val
        X[k + 1] = x.flatten()
        U[k] = u_val

        state_change = float(np.linalg.norm(X[k + 1] - X[k]))

        # Idle part of the control period: refill the Enc(0) pool (not timed,
        # not on the critical path; no-op in "baseline" mode).
        t5 = time.perf_counter()
        ctrl.idle()
        timings["offline"].append((time.perf_counter() - t5) * 1e3)

        trace.append(
            {
                "step": k,
                "time_s": time_s,
                "state_before": x_before.tolist(),   # full x(k), all n states
                "state_after": x.flatten().tolist(), # full x(k+1), all n states
                "u_encrypted": u_val,
                "u_plain_check": u_plain_check,
                "control_error": control_error,
                "state_change_norm": state_change,
                "encrypt_ms": encrypt_ms,
                "he_dot_ms": he_ms,
                "decrypt_ms": decrypt_ms,
                "enc_x_bytes": enc_x_bytes,
                "enc_u_bytes": enc_u_bytes,
            }
        )

        # ------------------------------------------------------
        # TERMINAL DEMONSTRATION
        # ------------------------------------------------------
        show = verbose and (k < demo_steps or (k + 1) % print_every == 0 or k == steps - 1)

        if show:
            print(f"\nSTEP {k:03d}   t = {time_s:5.2f} s")
            print(f"  Sensor state x(k)       = {_fmt_vector(x_before)}")
            print("  -> Encrypting x(k) with CKKS")
            print(f"  -> Ciphertext created    ({enc_x_bytes} bytes)" if enc_x_bytes >= 0
                  else "  -> Ciphertext created")
            print("  -> Controller: Enc(x) · (-K)")
            print(f"  -> Encrypted result      ({enc_u_bytes} bytes)" if enc_u_bytes >= 0
                  else "  -> Encrypted result created")
            print(f"  -> Decrypting control u  = {u_val:.9f}")
            print(f"  -> Plant update x(k+1)   = {_fmt_vector(x.flatten())}")
            print(
                f"  -> timing: encrypt={encrypt_ms:.3f} ms, "
                f"HE={he_ms:.3f} ms, decrypt={decrypt_ms:.3f} ms"
            )

    ctrl.close()

    if verbose:
        avg_encrypt = np.mean(timings["encrypt"])
        avg_he = np.mean(timings["he_dot"])
        avg_decrypt = np.mean(timings["decrypt"])

        print("\n" + "=" * 72)
        print("EXECUTION COMPLETE")
        print("=" * 72)
        print(f"Control cycles completed : {steps}")
        print(f"Final state              : {_fmt_vector(X[-1])}")
        print(f"Maximum |u|              : {np.max(np.abs(U)):.9f}")
        print("\nAverage timing per control cycle:")
        print(f"  Encryption             : {avg_encrypt:.4f} ms")
        print(f"  Homomorphic dot product: {avg_he:.4f} ms")
        print(f"  Decryption             : {avg_decrypt:.4f} ms")
        print(
            f"  TOTAL                  : "
            f"{avg_encrypt + avg_he + avg_decrypt:.4f} ms"
        )

    return X, U, timings, trace


if __name__ == "__main__":
    A, B, C, D = build_plant()
    K = design_state_feedback(A, B)
    Ad, Bd, Cd, Dd = discretize(A, B, C, D, dt=0.1)

    n = A.shape[0]
    x0 = np.zeros(n)
    x0[2 * (N_MASSES - 1)] = 1.0

    print("=" * 72)
    print(f"{N_MASSES}-MASS CHAIN ({n} STATES) + CKKS CONTROLLER")
    print("=" * 72)
    print(f"Initial state x(0) = {_fmt_vector(x0)}")
    print(f"State-feedback gain K = {_fmt_vector(K)}")
    print(f"Sampling time = 0.1 s")
    print("Building CKKS context...")

    ctx = make_context()

    X, U, timings, trace = simulate_encrypted(
        Ad, Bd, K, x0, steps=200, ctx=ctx, verbose=True
    )

    print("\nThis run demonstrates:")
    print("  plaintext state -> CKKS ciphertext -> HE control -> decrypted u")
    print("  -> plant update -> next state")
