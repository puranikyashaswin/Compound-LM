# COMPOUND-LM protocol evidence (toy scale)

_Real held-out measurements from 738,144-parameter runs on CPU. Toy scale; the protocol is identical at 200M._

## Two-seed baseline (A0/A1)

- seed 17 val_acc: 0.9407
- seed 23 val_acc: 0.9368
- seed spread: 0.0040 (PASS at gate <= 0.15)

## Compounding / capability-at-cost

Target held-out accuracy: 0.8182

| Run | Levers | Cost (FLOPs) | FLOP mult. | Wall-clock mult. | Overlap |
|---|---|---:|---:|---:|---:|
| baseline-s17 | baseline | 2.406e+10 | 1.000× | noisy | 1.000× |
| baseline-s23 | baseline | 2.596e+10 | 0.927× | noisy | 0.927× |
| optimizer-s17 | optimizer | 1.323e+10 | 1.819× | noisy | 1.021× |
| optimizer-s23 | optimizer | 1.351e+10 | 1.781× | noisy | 1.000× |
| architecture-s17 | architecture | 4.664e+10 | 0.516× | noisy | 1.028× |
| architecture-s23 | architecture | 4.797e+10 | 0.502× | noisy | 1.000× |
| compound-s17 | architecture, optimizer | 1.658e+10 | 1.451× | noisy | 1.624× |
| compound-s23 | architecture, optimizer | 1.657e+10 | 1.452× | noisy | 1.625× |

_held-out intrinsic scores; real measurements, toy scale. Replace corpus/config/eval-harness to scale to the 200M baseline._

