"""Quick correctness checks.   python tests/test_correctness.py   (from repo root)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "controller_sim"))
import numpy as np
from state_space_model import build_plant, design_state_feedback, discretize, N_MASSES
from plaintext_sim import simulate_plaintext
from encrypted_sim import simulate_encrypted, make_context
from ckks_fast import EncZeroPool

A, B, C, D = build_plant(); K = design_state_feedback(A, B)
Ad, Bd, _, _ = discretize(A, B, C, D, dt=0.1)
x0 = np.zeros(18); x0[2 * (N_MASSES - 1)] = 1.0

ctx = make_context()
x = np.random.randn(18)
pool = EncZeroPool(ctx, 18, target=2); pool.fill()
err = np.max(np.abs(np.array((pool.pop() + x.tolist()).decrypt()) - x))
assert err < 1e-4, err; print(f"PASS  Enc(0)+x decrypts to x (err {err:.1e})")
assert pool.pop().serialize() != pool.pop().serialize(); print("PASS  pooled Enc(0) are distinct")

Xp, Up = simulate_plaintext(Ad, Bd, K, x0, 40)
for mode, bits, thr in [("baseline", (60,40,40,60), False), ("baseline", (60,40,60), False),
                        ("precomp", (60,40,60), False), ("precomp", (60,40,60), True),
                        ("norot", (60,40,60), False)]:
    c = make_context(coeff_mod_bit_sizes=bits, galois=(mode != "norot"))
    X, U, t, tr = simulate_encrypted(Ad, Bd, K, x0, 40, ctx=c, mode=mode, verbose=False,
                                     pool_threaded=thr)
    ue, xe = np.max(np.abs(U - Up)), np.max(np.linalg.norm(X - Xp, axis=1))
    assert ue < 1e-3 and xe < 1e-4 and len(tr[0]["state_before"]) == 18, (mode, ue, xe)
    print(f"PASS  {mode:8s} {str(bits):15s} threaded={thr!s:5s} u_err {ue:.1e} x_err {xe:.1e}")
print("ALL TESTS PASSED")
