# Encrypted State-Feedback Control Simulation

Software simulation of a state-feedback controller executing on homomorphically encrypted data.

## Overview

This project demonstrates encrypted state-feedback control using CKKS (Cheon-Kim-Kim-Song) homomorphic encryption.

The basic control law is:

    u = -Kx

where `x` is the system state and `K` is the state-feedback gain.

In the encrypted version, the state is encrypted before the controller computation:

    Enc(x) · (-K) = Enc(u)

**Plant: a 9-mass spring-damper chain (18 states)**, single actuator applied to the last
mass, single sensor measuring the last mass's position. This is the same architecture as
the original 2-state mass-spring-damper demo -- it's still a single-input, single-output
(SISO) controller with one gain row `K` and one HE dot product per control step -- just
with 9 masses daisy-chained together instead of 1, so the state vector grew from 2 to 18
without changing the control/encryption architecture itself.

    wall --[k,c]-- m1 --[k,c]-- m2 --[k,c]-- ... --[k,c]-- m9 <-- u(t)

The simulation includes:

- A 9-mass chain state-space plant model (18-state system)
- LQR state-feedback gain calculation (see "Why LQR?" below)
- Plaintext controller as a reference
- Encrypted controller using TenSEAL / CKKS
- Correctness and timing analysis
- Comparison plots

## Prerequisites

- Python 3.7+
- `pip` or equivalent package manager

## Installation

Install the required packages:

```bash
pip install tenseal scipy matplotlib numpy
```

If `pip` is not recognized:

```bash
python -m pip install tenseal scipy matplotlib numpy
```

## Running

From the project root:

```bash
cd controller_sim
```

### Run the complete simulation

```bash
python compare_and_plot.py
```

or:

```bash
python3 compare_and_plot.py
```

This runs the plaintext and encrypted simulations, compares their results, and generates the plots and numerical summary in `results/`.

### Run only the encrypted controller

To directly see the CKKS encrypted controller execution:

```bash
python encrypted_sim.py
```

or:

```bash
python3 encrypted_sim.py
```

This prints the execution of each control step, including:

- State being encrypted
- CKKS encryption
- Homomorphic computation `Enc(x) · (-K)`
- Encrypted control result
- Decryption
- Control value applied to the plant
- Encryption / HE / decryption timing

The plaintext `u = -Kx` calculation is also performed internally as a correctness check, but it is not used to generate the encrypted result or update the plant.

## Project structure

```
controller_sim/
├── state_space_model.py    # Plant model (9-mass chain) and controller gain
├── plaintext_sim.py        # Baseline (unencrypted) controller
├── encrypted_sim.py        # CKKS encrypted controller
├── compare_and_plot.py     # Run both and generate plots
├── ckks_fast.py            # Fast CKKS execution (Enc(0) pool, modes)
└── benchmark.py            # Ablation: which optimisation buys what

results/
├── state_response.png      # State trajectory comparison (first/middle/last mass)
├── control_signal.png      # Control input over time
├── error_vs_step.png       # HE approximation error (full 18-state norm)
├── timing_breakdown.png    # Operation timing
└── summary.txt             # Numerical results
```

## How it works

Each control step follows this flow:

```
Simulated state x   (18-dimensional)
    ↓
CKKS encryption
    ↓
Encrypted state Enc(x)
    ↓
Homomorphic computation
Enc(x) · (-K)
    ↓
Encrypted control Enc(u)   (still a single scalar u)
    ↓
Decrypt
    ↓
Control u
    ↓
Update plant
    ↓
Next state
```

The plaintext controller is used separately as a reference so that the decrypted HE result can be checked for correctness.

The actual control signal used to update the encrypted simulation comes from the decrypted homomorphic result.

## From 2 states to 18 states: what actually changed

Going from 2 states to 18 did **not** require changing the encryption logic. TenSEAL's
`enc_x.dot(neg_K)` already does a generic rotate-and-sum reduction over however many
slots the ciphertext holds -- it works identically for a 2-element or an 18-element
vector. `encrypted_sim.py` and `plaintext_sim.py` are unchanged in their control logic.

What changed:

1. **`state_space_model.py`** -- `build_plant()` now builds an 18x18 `A` and 18x1 `B`
   for a chain of 9 masses (each mass = 2 states: position, velocity) instead of a
   single 2x2 mass-spring-damper. The chain is still driven by exactly one actuator
   (on the last mass) and measured at exactly one sensor (also the last mass), so the
   controller stays single-input, single-output.
2. **Gain design** -- switched from manual pole placement to **LQR**. See below.
3. **Logging / plots** -- the per-step trace and CSV columns now cover all 18 states
   instead of two hardcoded ones, and the state-response plot shows 3 representative
   masses (first, middle, last) along the chain instead of a single position/velocity
   pair, so the plot stays readable instead of overlaying 9 lines.

### Why LQR instead of manual pole placement?

The original 2-state demo picked two closed-loop poles by hand (`-4, -5`) and used
`scipy.signal.place_poles`. For an 18-state single-actuator chain this doesn't scale
well: some vibration modes of a long chain are only weakly reachable from a single
actuator at one end, so forcing all 18 poles to specific hand-picked locations demands
enormous feedback gains (we measured gains in the hundreds of millions), which then
wrecks the CKKS approximation. LQR (`scipy.linalg.solve_continuous_are`) instead
balances control effort against state error and produces a much smaller, better-scaled
gain matrix (`max|K| ≈ 57` here) while still stabilizing the whole chain -- it's the
same `u = -Kx` control law, just designed differently.

## Why CKKS?

BFV is designed for exact modular integer arithmetic, while CKKS supports approximate arithmetic on real and complex values.

The state-feedback controller uses real-valued states and gains. Therefore, CKKS is more suitable for this simulation.

TenSEAL provides the Python interface used here, with Microsoft SEAL as the underlying homomorphic encryption library.

## Notes

- Each control step encrypts the current state independently.
- The simulation does not repeatedly operate on the same ciphertext across control steps, so the 200 control steps should not be interpreted as cumulative noise growth in a single ciphertext.
- CKKS performs approximate arithmetic, so a small difference between the decrypted HE result and the plaintext reference is expected (measured here: max control error ~2e-5).
- The approximation error is measured directly from the simulation rather than assuming a fixed error range.
- The plant dynamics are simulated in plaintext; the state-feedback controller computation is the part performed homomorphically.
- Runtime depends on the CKKS parameters. Larger polynomial degrees and coefficient moduli can increase computational cost.
- The number of masses is a single constant (`N_MASSES` in `state_space_model.py`) -- set it to 8 for a 16-state system, 9 for 18-state (default here), etc.

## Current scope

This is a software proof-of-concept for the encrypted controller.

The current implementation demonstrates:

- State-space modelling (9-mass chain, 18 states)
- State-feedback control (LQR design)
- CKKS encryption
- Homomorphic controller computation
- Decryption of the control signal
- Correctness comparison
- Timing analysis

FPGA implementation and hardware-level optimizations such as NTT acceleration, pipelining, parallelism, and resource optimization are outside the scope of this simulation.

## Faster CKKS execution (added)

`python compare_and_plot.py` now runs a faster path by default. **Every output file, CSV header, plot and
terminal/summary format is unchanged**; the plant (`state_space_model.py`) and plaintext reference
(`plaintext_sim.py`) are untouched. Only the inside of the encrypted step changed.

```bash
python compare_and_plot.py                                   # fast, fully homomorphic (default)
python compare_and_plot.py --mode baseline --paper-params    # exactly the original behaviour
python compare_and_plot.py --mode norot                      # fastest; see trade-off below
python compare_and_plot.py --threaded-pool                   # Enc(0) generated on background threads
python benchmark.py --repeats 5                              # ablation table + chart in results/
python ../tests/test_correctness.py
```

| change | why it is faster | effect on results |
|---|---|---|
| modulus chain `[60,40,60]` (was `[60,40,40,60]`) | the dot product uses one level, the 2nd 40-bit prime was unused; fewer RNS limbs make every rotation cheaper | none (same N=8192, 128-bit security, scale 2^40, same error) |
| Enc(0) pre-computation | CKKS encryption is `Enc(0) + Encode(x)`; `Enc(0)` is independent of `x(k)`, so it is generated in the idle part of the 100 ms period (optionally on threads) and used once | none (standard encryption) |
| serialisation outside timers | sizes are still recorded; they no longer inflate encrypt/decrypt time | timing columns only |

Measured (200 steps, same machine, interleaved runs): original **20.95 ms/step** -> new default **8.04 ms/step** (2.6x).
Accuracy is unchanged (max state error ~1.4e-6, max control error ~2.2e-5).

Notes:
* Timings are **critical-path** latency. The Enc(0) generation (~4.4 ms/step) still costs CPU, but it happens
  off the critical path. Total work per step is ~12.4 ms versus ~20.2 ms originally.
* Control steps cannot be parallelised: `x(k+1)` needs `u(k)`. Only the Enc(0) pool is independent of the state.
* `--mode norot` skips the rotate-and-sum: the controller returns the 18 products `K_i*x_i` and the key
  holder sums them. About 7x faster, but the key holder can then recover `K`. Opt-in only.
* The ciphertexts are smaller with the new modulus (x: 235 KB, u: 131 KB, previously 331 KB / 235 KB), so
  the `enc_*_serialized_bytes` columns change.
* The `--mode`/`--paper-params` flags are recorded in `results/summary.txt`.
