# `tri_shield_core` — EGM Component Contract ("Peanuts")

## Purpose
Provides zero-allocation, lock-free sub-microsecond safety and compute filters (`NeuromorphicShield`, `HamiltonianShield`, `WalJepaBuffer`, `CbfPreTradeGateway`, and `ProductionTradeoffSupervisor`) for end-to-end neural world models.

## File Contracts
- [`src/lib.rs`](src/lib.rs):
  - `NeuromorphicShield`: Biological retina log-luminance event gate (`|Δ ln I| >= C_thresh`) pruning static tokens + Leaky Integrate-and-Fire (LIF) sub-frame optical-flow $\tau$-divergence spinal reflex.
  - `HamiltonianShield`: Symplectic Leapfrog energy invariant (`|ΔH - W_ext| <= ε`) catching sensory dropouts and glare-induced teleportation.
  - `WalJepaBuffer`: Write-Ahead Log ring-buffer with Mahalanobis innovation verification against `1,000 Hz` IMU telemetry (`ROLLBACK TO SAVEPOINT`).
  - `CbfPreTradeGateway`: Analytical active-set Control Barrier Function Quadratic Program projecting neural actions onto headway and Kamm friction-circle boundaries.
  - `ProductionTradeoffSupervisor`: Real-world fleet feasibility supervisor enforcing `$0` BOM CMOS rolling-shutter line-buffer emulation, $16\times 16$ GPU warp tile coalescing (`78.5%` structured sparsity $\rightarrow$ `11.4×` real systolic speedup) with `1/16` keyframe refresh, `450 ms` Dissipative Port-Hamiltonian decay (`0.18%` phantom braking), straightaway zero-excitation conservative friction floor ($\mu_{\text{floor}} = 0.45$), and `60 ms` hydraulic/EPAS actuator delay + jerk slew-rate High-Order CBF (`HOCBF`).
