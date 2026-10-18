# Run the improved simulation

From the `controller_sim` directory:

```bash
python encrypted_sim.py
```

This runs the CKKS encrypted controller and prints the execution pipeline:
sensor state -> CKKS encryption -> homomorphic dot product -> decryption -> plant update.

Then run:

```bash
python compare_and_plot.py
```

This runs plaintext and encrypted controllers, compares them, and creates:

- `../results/control_data.csv` — numerical dataset
- `../results/execution_trace.csv` — step-by-step encrypted execution data
- `../results/state_response.png`
- `../results/control_signal.png`
- `../results/error_vs_step.png`
- `../results/timing_breakdown.png`
- `../results/summary.txt`

The CSV files can be opened in Excel.

## Dependencies

```bash
pip install numpy scipy matplotlib tenseal
```

TenSEAL must be installed in the Python environment used to run the scripts.
