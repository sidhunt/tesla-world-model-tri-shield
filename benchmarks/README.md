# `benchmarks` — EGM Component Contract ("Peanuts")

## Purpose
Executes paired 4-arm Monte Carlo Out-Of-Distribution (OOD) falsification experiments comparing unshielded neural world models against incremental shield combinations.

## File Contracts
- [`run_ablation_suite.py`](run_ablation_suite.py): Runs `10,000` seeded trials per arm (`40,000` total) across sun-glare occlusion, black-ice friction collapse, and sudden cut-in scenarios. Writes structured output to [`results.json`](results.json).
- [`results.json`](results.json): Ground-truth benchmark output containing collision rates, spinout rates, total failure rates, and microsecond latency measurements.
