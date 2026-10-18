"""
============================================================
ABLATION BENCHMARK: which optimisation buys how much?  (18-state chain)
============================================================
Runs the same 18-state closed loop (200 steps) for every configuration,
repeated --repeats times (fresh context each time), and reports:

  * median critical-path latency per step (encrypt + HE + decrypt)
  * offline work moved off the critical path (Enc(0) generation)
  * max |u_enc - u_plain|, max ||x_enc - x_plain||_2   (accuracy)

    python benchmark.py                # 5 repeats
    python benchmark.py --repeats 10

Outputs: ../results/benchmark_ablation.csv / .md / .png
============================================================
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from state_space_model import build_plant, design_state_feedback, discretize, N_MASSES
from plaintext_sim import simulate_plaintext
from encrypted_sim import simulate_encrypted, make_context

DT = 0.1
PAPER = (60, 40, 40, 60)
LEAN = (60, 40, 60)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# name, mode, profile, keeps K hidden from key holder?, description
CONFIGS = [
    ("1 baseline (original)",      "baseline", PAPER, True,  "ts.ckks_vector + dot, [60,40,40,60]"),
    ("2 + lean modulus",           "baseline", LEAN,  True,  "same, [60,40,60]"),
    ("3 + Enc(0) pre-computation", "precomp",  PAPER, True,  "Enc(0) pool, [60,40,40,60]"),
    ("4 pre-comp + lean  (SAFE)",  "precomp",  LEAN,  True,  "full HE dot, K stays private"),
    ("5 no-rotation, paper",       "norot",    PAPER, False, "slot-sum by key holder"),
    ("6 no-rotation + lean (MAX)", "norot",    LEAN,  False, "fastest; K exposed to key holder"),
]


def run(repeats, steps):
    A, B, C, D = build_plant()
    K = design_state_feedback(A, B)
    Ad, Bd, _, _ = discretize(A, B, C, D, dt=DT)
    x0 = np.zeros(A.shape[0]); x0[2 * (N_MASSES - 1)] = 1.0
    Xp, Up = simulate_plaintext(Ad, Bd, K, x0, steps)
    rows = []
    for name, mode, prof, k_private, desc in CONFIGS:
        crit, enc, he, dec, off, cerr, serr = [], [], [], [], [], [], []
        for r in range(repeats):
            ctx = make_context(coeff_mod_bit_sizes=prof, galois=(mode != "norot"))
            X, U, t, _ = simulate_encrypted(Ad, Bd, K, x0, steps, ctx=ctx, mode=mode,
                                            verbose=False)
            # drop first 5 steps (cache / allocator warm-up)
            w = slice(5, None)
            enc.append(np.mean(t["encrypt"][w])); he.append(np.mean(t["he_dot"][w]))
            dec.append(np.mean(t["decrypt"][w])); off.append(np.mean(t["offline"][w]))
            crit.append(enc[-1] + he[-1] + dec[-1])
            cerr.append(np.max(np.abs(U - Up)))
            serr.append(np.max(np.linalg.norm(X - Xp, axis=1)))
        rows.append(dict(config=name, mode=mode, profile=str(prof), k_private=k_private,
                         description=desc,
                         encrypt_ms=np.median(enc), he_ms=np.median(he),
                         decrypt_ms=np.median(dec), critical_ms=np.median(crit),
                         critical_min_ms=np.min(crit), critical_max_ms=np.max(crit),
                         offline_ms=np.median(off),
                         max_control_error=np.max(cerr), max_state_error_l2=np.max(serr)))
        print(f"{name:32s} crit {rows[-1]['critical_ms']:7.3f} ms  "
              f"(enc {rows[-1]['encrypt_ms']:.2f} he {rows[-1]['he_ms']:.2f} "
              f"dec {rows[-1]['decrypt_ms']:.2f})  offline {rows[-1]['offline_ms']:.2f}  "
              f"u_err {rows[-1]['max_control_error']:.2e}  x_err {rows[-1]['max_state_error_l2']:.2e}")
    return rows


def write(rows, repeats, steps):
    base = rows[0]["critical_ms"]
    with open(os.path.join(RESULTS, "benchmark_ablation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["speedup_vs_baseline"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "speedup_vs_baseline": base / r["critical_ms"]})
    with open(os.path.join(RESULTS, "benchmark_ablation.md"), "w") as f:
        f.write(f"CKKS ablation, 18 states, {steps} steps, median of {repeats} runs\n\n")
        f.write("| configuration | enc | HE | dec | critical ms/step | speed-up | "
                "offline ms | max u err | max x err | K private |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['config']} | {r['encrypt_ms']:.2f} | {r['he_ms']:.2f} | "
                    f"{r['decrypt_ms']:.2f} | **{r['critical_ms']:.2f}** | "
                    f"{base/r['critical_ms']:.1f}x | {r['offline_ms']:.2f} | "
                    f"{r['max_control_error']:.2e} | {r['max_state_error_l2']:.2e} | "
                    f"{'yes' if r['k_private'] else 'NO'} |\n")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [r["config"] for r in rows]
    y = np.arange(len(rows))
    e = [r["encrypt_ms"] for r in rows]; h = [r["he_ms"] for r in rows]
    d = [r["decrypt_ms"] for r in rows]
    ax.barh(y, e, label="encrypt (online)", color="#4472C4")
    ax.barh(y, h, left=e, label="HE evaluation", color="#ED7D31")
    ax.barh(y, d, left=np.add(e, h), label="decrypt", color="#70AD47")
    for i, r in enumerate(rows):
        ax.text(r["critical_ms"] + 0.2, i, f"{r['critical_ms']:.2f} ms  ({base/r['critical_ms']:.1f}x)",
                va="center", fontsize=9)
    ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set_xlabel("critical-path latency per control step (ms)")
    ax.set_title("CKKS controller: latency ablation (18 states)")
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=.3)
    ax.set_xlim(0, max(r["critical_ms"] for r in rows) * 1.25)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "benchmark_ablation.png"), dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()
    rows = run(a.repeats, a.steps)
    write(rows, a.repeats, a.steps)
    print("\nwrote results/benchmark_ablation.{csv,md,png}")
