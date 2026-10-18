"""
============================================================
COMPARE PLAINTEXT vs ENCRYPTED CONTROLLER
============================================================

This file runs both controllers and produces the same set of
deliverables as the original 2-state version:

    results/
        control_data.csv
        execution_trace.csv
        state_response.png
        control_signal.png
        error_vs_step.png
        timing_breakdown.png
        summary.txt

The only things that changed for the 18-state upgrade are:
  - build_plant()/design_state_feedback() now return an 18x18 /
    1x18 system instead of 2x2 / 1x2 (see state_space_model.py).
  - CSV columns and the state_response plot are generated in a loop
    over all masses instead of being hardcoded to "position"/
    "velocity", since there are now N_MASSES of each.
  - state_response.png shows 3 representative masses (first, middle,
    last) along the chain instead of just "the" mass, so the plot
    stays readable instead of cramming 9 overlapping lines together.
Everything else (control_signal.png, error_vs_step.png,
timing_breakdown.png, the CSV writer, the overall flow) is unchanged.
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


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_csv(path, headers, rows):
    """Write a simple CSV file that can be opened in Excel."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="precomp", choices=["baseline", "precomp", "norot"],
                    help="baseline = original path; precomp = fast, fully homomorphic (default); "
                         "norot = fastest, key holder sums slots (can recover K)")
    ap.add_argument("--paper-params", action="store_true",
                    help="use the original modulus chain [60,40,40,60] instead of [60,40,60]")
    ap.add_argument("--threaded-pool", action="store_true",
                    help="generate Enc(0) on background threads")
    ap.add_argument("--pool-workers", type=int, default=2)
    args = ap.parse_args()
    coeff_bits = (60, 40, 40, 60) if args.paper_params else (60, 40, 60)

    # ---------------------------------------------------------
    # BUILD PLANT + CONTROLLER
    # ---------------------------------------------------------
    A, B, C, D = build_plant()
    K = design_state_feedback(A, B)
    n = A.shape[0]
    dt = 0.1
    Ad, Bd, Cd, Dd = discretize(A, B, C, D, dt=dt)

    x0 = np.zeros(n)
    x0[2 * (N_MASSES - 1)] = 1.0   # displace the last mass by 1m, rest at 0
    steps = 200
    t = np.arange(steps + 1) * dt

    print("=" * 72)
    print(f"STATE-SPACE MODEL  ({N_MASSES} masses -> {n} states)")
    print("=" * 72)
    print("Open-loop poles:", np.linalg.eigvals(A))
    print("\nState feedback gain K:\n", K)
    print("Closed-loop poles (A-BK):\n", np.linalg.eigvals(A - B @ K))

    # ---------------------------------------------------------
    # PLAINTEXT BASELINE
    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("1) PLAINTEXT BASELINE")
    print("=" * 72)

    X_plain, U_plain = simulate_plaintext(Ad, Bd, K, x0, steps)

    print(f"Initial state : {X_plain[0]}")
    print(f"Final state   : {X_plain[-1]}")
    print(f"Max |u|       : {np.max(np.abs(U_plain)):.9f}")

    # ---------------------------------------------------------
    # ENCRYPTED RUN
    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("2) CKKS ENCRYPTED CONTROLLER")
    print("=" * 72)

    ctx = make_context(coeff_mod_bit_sizes=coeff_bits, galois=(args.mode != "norot"))
    X_enc, U_enc, timings, trace = simulate_encrypted(
        Ad, Bd, K, x0, steps, ctx=ctx, verbose=True,
        mode=args.mode, pool_threaded=args.threaded_pool,
        pool_workers=args.pool_workers,
    )

    # ---------------------------------------------------------
    # ACCURACY
    # ---------------------------------------------------------
    state_error = np.linalg.norm(X_plain - X_enc, axis=1)
    control_error = np.abs(U_plain - U_enc)

    max_state_error = np.max(state_error)
    max_control_error = np.max(control_error)

    print("\n" + "=" * 72)
    print("3) PLAINTEXT vs CKKS ACCURACY")
    print("=" * 72)
    print(f"Final plaintext state  : {X_plain[-1]}")
    print(f"Final encrypted state  : {X_enc[-1]}")
    print(f"Maximum state error    : {max_state_error:.6e}")
    print(f"Maximum control error  : {max_control_error:.6e}")
    print("\nInterpretation:")
    print("CKKS performs approximate arithmetic, so a small numerical")
    print("difference between plaintext and encrypted execution is expected.")

    # ---------------------------------------------------------
    # SAVE ACTUAL NUMERICAL DATA
    # ---------------------------------------------------------
    # Column headers built per-mass (pos/vel, plaintext + encrypted),
    # since there are now N_MASSES masses instead of just one.
    state_headers = []
    for i in range(N_MASSES):
        state_headers += [
            f"pos{i+1}_plain", f"vel{i+1}_plain",
            f"pos{i+1}_encrypted", f"vel{i+1}_encrypted",
        ]

    control_rows = []
    for k in range(steps):
        tr = trace[k]
        row = [k, f"{t[k]:.6f}"]
        for i in range(N_MASSES):
            row += [
                f"{X_plain[k, 2*i]:.12f}", f"{X_plain[k, 2*i+1]:.12f}",
                f"{X_enc[k, 2*i]:.12f}", f"{X_enc[k, 2*i+1]:.12f}",
            ]
        row += [
            f"{U_plain[k]:.12f}", f"{U_enc[k]:.12f}",
            f"{state_error[k]:.12e}", f"{control_error[k]:.12e}",
            f"{tr['encrypt_ms']:.6f}", f"{tr['he_dot_ms']:.6f}", f"{tr['decrypt_ms']:.6f}",
            tr["enc_x_bytes"], tr["enc_u_bytes"],
        ]
        control_rows.append(row)

    save_csv(
        os.path.join(RESULTS_DIR, "control_data.csv"),
        ["step", "time_s"] + state_headers + [
            "u_plain", "u_encrypted",
            "state_error", "control_error",
            "encrypt_ms", "he_dot_ms", "decrypt_ms",
            "enc_x_serialized_bytes", "enc_u_serialized_bytes",
        ],
        control_rows,
    )

    trace_state_headers = [f"x{i+1}" for i in range(n)]
    trace_rows = []
    for tr in trace:
        row = [tr["step"], f"{tr['time_s']:.6f}"]
        row += [f"{v:.12f}" for v in tr["state_before"]]
        row += [
            f"{tr['u_encrypted']:.12f}",
            f"{tr['u_plain_check']:.12f}",
            f"{tr['control_error']:.12e}",
        ]
        row += [f"{v:.12f}" for v in tr["state_after"]]
        row += [
            f"{tr['state_change_norm']:.12e}",
            f"{tr['encrypt_ms']:.6f}",
            f"{tr['he_dot_ms']:.6f}",
            f"{tr['decrypt_ms']:.6f}",
            tr["enc_x_bytes"],
            tr["enc_u_bytes"],
        ]
        trace_rows.append(row)

    save_csv(
        os.path.join(RESULTS_DIR, "execution_trace.csv"),
        (["step", "time_s"]
         + [f"sensor_{h}" for h in trace_state_headers]
         + ["u_from_encrypted_controller", "u_plain_validation", "control_error"]
         + [f"next_{h}" for h in trace_state_headers]
         + ["state_change_norm", "encrypt_ms", "he_dot_ms",
            "decrypt_ms", "enc_x_serialized_bytes", "enc_u_serialized_bytes"]),
        trace_rows,
    )

    # ---------------------------------------------------------
    # PLOTS
    # ---------------------------------------------------------
    # Show 3 representative masses along the chain (first, middle,
    # last) so the figure stays readable instead of overlaying
    # N_MASSES lines on top of each other.
    rep_indices = sorted(set([0, N_MASSES // 2, N_MASSES - 1]))
    rep_labels = {0: "Mass 1 (fixed end)", N_MASSES - 1: "Mass N (actuated end)"}

    fig, axs = plt.subplots(len(rep_indices), 1, figsize=(8, 3 * len(rep_indices)), sharex=True)
    if len(rep_indices) == 1:
        axs = [axs]

    for ax, i in zip(axs, rep_indices):
        label = rep_labels.get(i, f"Mass {i+1} (middle)")
        ax.plot(t, X_plain[:, 2*i], label="Plaintext controller", lw=2)
        ax.plot(t, X_enc[:, 2*i], "--", label="Encrypted controller (CKKS)", lw=2)
        ax.set_ylabel("Position (m)")
        ax.set_title(label)
        ax.legend()
        ax.grid(True)
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Closed-Loop State Response (representative masses along the chain)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "state_response.png"), dpi=150)
    plt.close(fig)

    # Control signal is still a single scalar u(k) -- this plot is
    # unchanged from the 2-state version.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t[:-1], U_plain, label="Plaintext u(k)", lw=2)
    ax.plot(t[:-1], U_enc, "--", label="Encrypted u(k)", lw=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control input u(k)")
    ax.set_title("Control Signal: Plaintext vs CKKS")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "control_signal.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(t, state_error + 1e-16)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State error, ||x_plain - x_enc|| (log scale)")
    ax.set_title(f"CKKS Approximation Error (full {n}-state vector norm)")
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "error_vs_step.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ops = ["encrypt", "he_dot", "decrypt"]
    means = [np.mean(timings[o]) for o in ops]
    ax.bar(ops, means)
    ax.set_ylabel("Average time per control step (ms)")
    ax.set_title(f"CKKS Controller Timing Breakdown ({n} states)")
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.3f} ms", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "timing_breakdown.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")

    avg_encrypt = np.mean(timings["encrypt"])
    avg_he = np.mean(timings["he_dot"])
    avg_decrypt = np.mean(timings["decrypt"])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("CKKS ENCRYPTED STATE-FEEDBACK CONTROL SYSTEM\n")
        f.write("=" * 72 + "\n\n")

        f.write("1. PLANT\n")
        f.write(f"{N_MASSES}-mass chain: mass 1 anchored to a wall, mass {N_MASSES}\n")
        f.write("driven by a single actuator; identical spring/damper between\n")
        f.write("each neighbouring pair (m=1 kg, c=1.0 Ns/m, k=2 N/m).\n")
        f.write(f"State vector (n={n}): [pos1, vel1, pos2, vel2, ..., pos{N_MASSES}, vel{N_MASSES}]\n")
        f.write(f"Open-loop poles:\n{np.linalg.eigvals(A)}\n\n")

        f.write("2. CONTROLLER\n")
        f.write("Control law: u(k) = -K x(k)   (single actuator, single gain row)\n")
        f.write("Gain design: LQR (solve_continuous_are), not manual pole placement --\n")
        f.write("a long single-actuator chain has very unevenly controllable modes,\n")
        f.write("so hand-picked poles blow up the gain magnitude and hurt CKKS accuracy.\n")
        f.write(f"K =\n{np.array2string(K, precision=4, max_line_width=100)}\n\n")
        f.write(f"Closed-loop poles (A-BK):\n{np.linalg.eigvals(A - B @ K)}\n\n")

        f.write("3. DISCRETIZATION\n")
        f.write(f"Sampling time: {dt} s\n")
        f.write(f"Ad shape: {Ad.shape}, Bd shape: {Bd.shape}\n\n")

        f.write("4. CKKS CONFIGURATION\n")
        f.write("Polynomial modulus degree: 8192\n")
        f.write(f"Coefficient modulus bits: {list(coeff_bits)}\n")
        f.write(f"Execution mode: {args.mode}\n")
        f.write("Global scale: 2^40\n")
        f.write("Library: TenSEAL (Microsoft SEAL backend)\n\n")

        f.write("5. SIMULATION\n")
        f.write(f"Control cycles: {steps}\n")
        f.write(f"Initial state: {x0}\n")
        f.write(f"Final plaintext state: {X_plain[-1]}\n")
        f.write(f"Final encrypted-controller state: {X_enc[-1]}\n\n")

        f.write("6. ACCURACY\n")
        f.write(f"Maximum state error (full {n}-dim vector norm): {max_state_error:.12e}\n")
        f.write(f"Maximum control error: {max_control_error:.12e}\n\n")

        f.write("7. HE TIMING\n")
        f.write(f"Average encryption: {avg_encrypt:.6f} ms/step\n")
        f.write(f"Average HE dot product: {avg_he:.6f} ms/step\n")
        f.write(f"Average decryption: {avg_decrypt:.6f} ms/step\n")
        f.write(f"Total: {avg_encrypt + avg_he + avg_decrypt:.6f} ms/step\n\n")

        f.write("8. FILES\n")
        f.write("control_data.csv      - combined numerical results (all masses)\n")
        f.write("execution_trace.csv   - step-by-step execution trace (all states)\n")
        f.write("state_response.png    - state comparison (first/middle/last mass)\n")
        f.write("control_signal.png    - control comparison\n")
        f.write("error_vs_step.png     - CKKS numerical error\n")
        f.write("timing_breakdown.png  - HE timing\n\n")

        f.write("9. INTERPRETATION\n")
        f.write("The controller computes the state-feedback dot product using\n")
        f.write("a CKKS ciphertext and a plaintext gain vector. Only the resulting\n")
        f.write("control signal is decrypted before it is applied to the simulated\n")
        f.write("plant. The plaintext controller is retained as a baseline so the\n")
        f.write("effect of CKKS approximation can be quantified. Note that the\n")
        f.write("HE pipeline itself (encrypt -> homomorphic dot product -> decrypt)\n")
        f.write("is unchanged from the 2-state version: TenSEAL's dot() already\n")
        f.write(f"handles a {n}-element vector the same way it handled a 2-element one.\n")

    print("\n" + "=" * 72)
    print("FILES GENERATED")
    print("=" * 72)
    print(f"Results directory: {os.path.abspath(RESULTS_DIR)}")
    print("  control_data.csv")
    print("  execution_trace.csv")
    print("  state_response.png")
    print("  control_signal.png")
    print("  error_vs_step.png")
    print("  timing_breakdown.png")
    print("  summary.txt")


if __name__ == "__main__":
    main()
