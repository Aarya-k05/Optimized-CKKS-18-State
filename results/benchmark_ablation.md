CKKS ablation, 18 states, 200 steps, median of 5 runs

| configuration | enc | HE | dec | critical ms/step | speed-up | offline ms | max u err | max x err | K private |
|---|---|---|---|---|---|---|---|---|---|
| 1 baseline (original) | 5.73 | 13.32 | 1.13 | **20.20** | 1.0x | 0.00 | 2.20e-05 | 1.38e-06 | yes |
| 2 + lean modulus | 4.85 | 7.47 | 0.53 | **12.84** | 1.6x | 0.00 | 2.20e-05 | 1.37e-06 | yes |
| 3 + Enc(0) pre-computation | 0.97 | 13.18 | 1.10 | **15.26** | 1.3x | 5.43 | 2.28e-05 | 1.35e-06 | yes |
| 4 pre-comp + lean  (SAFE) | 0.71 | 6.79 | 0.49 | **8.02** | 2.5x | 4.38 | 2.38e-05 | 1.39e-06 | yes |
| 5 no-rotation, paper | 0.95 | 1.87 | 1.07 | **3.89** | 5.2x | 5.63 | 2.23e-05 | 1.39e-06 | NO |
| 6 no-rotation + lean (MAX) | 0.78 | 1.45 | 0.51 | **2.75** | 7.4x | 4.99 | 2.13e-05 | 1.36e-06 | NO |
