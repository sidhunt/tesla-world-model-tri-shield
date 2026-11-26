# `tri_shield_core` — EGM Component Contract ("Peanuts")

## Purpose
Provides zero-allocation, deterministic Rust supervisors for latent world models operating at 1,000 Hz.

## File Contracts & Invariants
- [`src/lib.rs`](src/lib.rs):
  - `LatentState::hamiltonian(&self) -> f64`: Computes total mechanical energy $H(q,p) = \frac{\|p\|^2}{2m} + \frac{1}{2}k\|q\|^2$.
  - `HamiltonianShield::verify_transition(...) -> (bool, f64, LatentState)`: Propagates state via Symplectic Velocity-Verlet and rejects transitions where $|\Delta H| > \epsilon_{\text{max}}$.
  - `WalJepaBuffer::verify_and_commit(&mut self, measured_p: [f64; 2]) -> (bool, f64, LatentState)`: Evaluates Mahalanobis residual $d_M$ against physical IMU momentum; commits checkpoint or aborts speculative ring buffer.
  - `CbfPreTradeGateway::filter_control(...) -> (ControlCommand, bool, f64)`: Enforces longitudinal ZCBF headway $h(x) \ge 0$ and radial Kamm friction circle $a_x^2 + a_y^2 \le (\mu g)^2$.
