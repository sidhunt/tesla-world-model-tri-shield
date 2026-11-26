# `benchmarks` — EGM Component Contract ("Peanuts")

## Purpose
Executes paired 5-arm Monte Carlo Out-Of-Distribution (OOD) falsification experiments and external industry/academic leaderboard validations (`Comma.ai Controls Challenge v2 TinyPhysics ONNX`, `NAVSIM v1/v2 PDMS & EPDMS`, `Bench2Drive DS & SR %`).

## File Contracts
- [`run_ablation_suite.py`](run_ablation_suite.py): Runs `10,000` seeded trials per arm (`50,000` total) across sun-glare occlusion, black-ice friction collapse, sudden cut-in, and sub-frame inter-frame pop-out scenarios. Writes structured output to [`results.json`](results.json).
- [`results.json`](results.json): Ground-truth 5-arm ablation output containing collision rates, spinout rates, total failure rates, neuromorphic token sparsity (`92.4%`), attention FLOPs (`0.58%`), and microsecond latency measurements (`1.251 µs/step`).
- [`run_external_leaderboards.py`](run_external_leaderboards.py): Evaluates the causal `quad_shield` controller on Comma.ai's `tinyphysics.onnx` autoregressive World Model across 100 real `openpilot` highway segments from `commaai/commaSteeringControl` and computes official NAVSIM v1 (`PDMS`), NAVSIM v2 (`EPDMS`), and Bench2Drive (`DS` / `SR %`) closed-loop leaderboard metrics. Writes verified output to [`external_leaderboards.json`](external_leaderboards.json).
- [`external_leaderboards.json`](external_leaderboards.json): Ground-truth external leaderboard comparison scores (`78.813` vs. `104.032` PID on Comma.ai TinyPhysics; `96.29 PDMS` / `85.70 EPDMS` on NAVSIM; `94.07 DS` on Bench2Drive).
