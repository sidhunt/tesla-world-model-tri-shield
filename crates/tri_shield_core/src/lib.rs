//! Tri-Shield Latent World Model Supervisor (`tri_shield_core`)
//!
//! Unifies three cross-industry fault-tolerant mechanisms into a sub-millisecond
//! supervisor between an End-to-End Neural World Model and vehicle/rocket actuators:
//! 1. Shield 1 (Airbus / SpaceX): Symplectic Hamiltonian Energy Invariant Filter
//! 2. Shield 2 (PostgreSQL WAL + tokencut): Two-Stage Speculative Rollout & Mahalanobis IMU Rollback
//! 3. Shield 3 (Wall Street HFT): Pre-Actuation Control Barrier Function (CBF) Active-Set QP Gateway

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatentState {
    /// Generalized coordinates q = (x, y) in meters (or latent metric frame)
    pub q: [f64; 2],
    /// Conjugate momenta p = (p_x, p_y) in kg*m/s
    pub p: [f64; 2],
    /// Effective mass (kg)
    pub mass: f64,
    /// Stiffness / potential field coefficient k
    pub potential_k: f64,
}

impl LatentState {
    /// Total Hamiltonian H(q, p) = T(p) + V(q)
    #[inline]
    pub fn hamiltonian(&self) -> f64 {
        let p_sq = self.p[0] * self.p[0] + self.p[1] * self.p[1];
        let q_sq = self.q[0] * self.q[0] + self.q[1] * self.q[1];
        let kinetic = p_sq / (2.0 * self.mass.max(1e-6));
        let potential = 0.5 * self.potential_k * q_sq;
        kinetic + potential
    }
}

/// Shield 1: Airbus Fly-By-Wire / SpaceX Symplectic Hamiltonian Envelope Protection
#[derive(Debug, Clone, Copy)]
pub struct HamiltonianShield {
    pub max_energy_drift: f64,
    pub dt: f64,
}

impl HamiltonianShield {
    /// Symplectic Velocity-Verlet / Leapfrog propagation step
    #[inline]
    pub fn symplectic_step(&self, state: &LatentState, external_force: [f64; 2]) -> LatentState {
        let dt = self.dt;
        let grad_v0 = [state.potential_k * state.q[0], state.potential_k * state.q[1]];
        let p_half = [
            state.p[0] + 0.5 * dt * (external_force[0] - grad_v0[0]),
            state.p[1] + 0.5 * dt * (external_force[1] - grad_v0[1]),
        ];
        let q_next = [
            state.q[0] + dt * (p_half[0] / state.mass),
            state.q[1] + dt * (p_half[1] / state.mass),
        ];
        let grad_v1 = [state.potential_k * q_next[0], state.potential_k * q_next[1]];
        let p_next = [
            p_half[0] + 0.5 * dt * (external_force[0] - grad_v1[0]),
            p_half[1] + 0.5 * dt * (external_force[1] - grad_v1[1]),
        ];
        LatentState {
            q: q_next,
            p: p_next,
            mass: state.mass,
            potential_k: state.potential_k,
        }
    }

    /// Evaluates whether a neural World Model's predicted latent state violates Hamiltonian conservation.
    /// If violated (e.g., object vanishes in sun glare or teleports), returns (false, drift, symplectic_fallback).
    #[inline]
    pub fn verify_transition(
        &self,
        prev: &LatentState,
        neural_pred: &LatentState,
        external_force: [f64; 2],
    ) -> (bool, f64, LatentState) {
        let symplectic_ref = self.symplectic_step(prev, external_force);
        let h_expected = symplectic_ref.hamiltonian();
        let h_pred = neural_pred.hamiltonian();
        let drift = (h_pred - h_expected).abs();
        if drift > self.max_energy_drift {
            (false, drift, symplectic_ref)
        } else {
            (true, drift, *neural_pred)
        }
    }
}

/// Shield 2: PostgreSQL Write-Ahead Log (WAL) + tokencut Two-Stage Telemetry Verifier
#[derive(Debug, Clone)]
pub struct WalJepaBuffer {
    pub committed_checkpoint: LatentState,
    pub speculative_wal: Vec<LatentState>,
    pub mahalanobis_threshold: f64,
    pub inv_cov_diag: [f64; 2],
}

impl WalJepaBuffer {
    pub fn new(initial: LatentState, threshold: f64, inv_cov_diag: [f64; 2]) -> Self {
        Self {
            committed_checkpoint: initial,
            speculative_wal: Vec::with_capacity(64),
            mahalanobis_threshold: threshold,
            inv_cov_diag,
        }
    }

    /// Stage 1: Append speculative rollout step to uncommitted WAL
    #[inline]
    pub fn append_speculative(&mut self, state: LatentState) {
        self.speculative_wal.push(state);
    }

    /// Stage 2: Deep verifier against 1,000 Hz physical IMU/odometry momentum measurement
    #[inline]
    pub fn verify_and_commit(&mut self, measured_p: [f64; 2]) -> (bool, f64, LatentState) {
        let candidate = match self.speculative_wal.last() {
            Some(c) => *c,
            None => return (true, 0.0, self.committed_checkpoint),
        };
        let dp0 = measured_p[0] - candidate.p[0];
        let dp1 = measured_p[1] - candidate.p[1];
        let d_m = (dp0 * dp0 * self.inv_cov_diag[0] + dp1 * dp1 * self.inv_cov_diag[1]).sqrt();

        if d_m > self.mahalanobis_threshold {
            // Atomic Transaction Abort: flush corrupted speculative dream & rollback
            self.speculative_wal.clear();
            (false, d_m, self.committed_checkpoint)
        } else {
            self.committed_checkpoint = candidate;
            self.speculative_wal.clear();
            (true, d_m, self.committed_checkpoint)
        }
    }
}

/// Shield 3: Wall Street HFT Pre-Actuation Control Barrier Function (CBF) Active-Set QP Gateway
#[derive(Debug, Clone, Copy)]
pub struct CbfPreTradeGateway {
    /// Minimum safe headway time T_h (seconds)
    pub headway_time: f64,
    /// Barrier decay rate alpha > 0
    pub alpha: f64,
    /// Road friction coefficient mu
    pub friction_mu: f64,
    /// Gravitational acceleration g (9.81 m/s^2)
    pub g: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlCommand {
    /// Longitudinal acceleration a_x (m/s^2)
    pub a_x: f64,
    /// Lateral acceleration a_y (m/s^2)
    pub a_y: f64,
}

impl CbfPreTradeGateway {
    /// Closed-form Active-Set Quadratic Program (QP) projection:
    /// 1. Enforces Zeroing CBF: h(x) = d_rel - T_h * v_ego >= 0
    ///    Lie derivative constraint: v_rel - T_h * a_x + alpha * h(x) >= 0
    ///    => a_x <= (v_rel + alpha * h(x)) / T_h
    /// 2. Enforces Kamm Friction Circle: a_x^2 + a_y^2 <= (mu * g)^2
    #[inline]
    pub fn filter_control(
        &self,
        u_nom: ControlCommand,
        d_rel: f64,
        v_ego: f64,
        v_lead: f64,
    ) -> (ControlCommand, bool, f64) {
        let h = d_rel - self.headway_time * v_ego;
        let v_rel = v_lead - v_ego;
        let max_safe_ax = (v_rel + self.alpha * h) / self.headway_time.max(0.1);

        let mut a_x = u_nom.a_x;
        let mut a_y = u_nom.a_y;
        let mut intervened = false;

        // 1. Apply CBF longitudinal barrier
        if a_x > max_safe_ax {
            a_x = max_safe_ax;
            intervened = true;
        }

        // 2. Project onto Kamm friction circle boundary
        let max_acc = self.friction_mu * self.g;
        let norm = (a_x * a_x + a_y * a_y).sqrt();
        if norm > max_acc {
            let scale = max_acc / norm.max(1e-6);
            a_x *= scale;
            a_y *= scale;
            intervened = true;
        }

        (ControlCommand { a_x, a_y }, intervened, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shield1_hamiltonian_catches_glare_teleportation() {
        let shield = HamiltonianShield {
            max_energy_drift: 50.0,
            dt: 0.01,
        };
        let prev = LatentState {
            q: [10.0, 0.0],
            p: [20000.0, 0.0], // 1000 kg vehicle at 20 m/s
            mass: 1000.0,
            potential_k: 10.0,
        };
        // Hallucinated state: lead vehicle vanishes / teleports in glare
        let hallucinated = LatentState {
            q: [95.0, 0.0],
            p: [35000.0, 0.0],
            mass: 1000.0,
            potential_k: 10.0,
        };
        let (ok, drift, fallback) = shield.verify_transition(&prev, &hallucinated, [0.0, 0.0]);
        assert!(!ok, "Shield 1 must reject OOD energy jump");
        assert!(drift > 50.0);
        assert!((fallback.q[0] - 10.2).abs() < 0.05);
    }

    #[test]
    fn test_shield2_wal_rollback_on_black_ice_slip() {
        let init = LatentState {
            q: [0.0, 0.0],
            p: [15000.0, 0.0],
            mass: 1000.0,
            potential_k: 1.0,
        };
        let mut wal = WalJepaBuffer::new(init, 3.0, [0.001, 0.001]);
        // Speculative dream assumes dry-road cornering momentum
        wal.append_speculative(LatentState {
            q: [1.0, 0.5],
            p: [15000.0, 4000.0],
            mass: 1000.0,
            potential_k: 1.0,
        });
        // Real IMU on black ice shows vehicle sliding straight (p_y = 100.0)
        let (committed, d_m, state) = wal.verify_and_commit([15000.0, 100.0]);
        assert!(!committed, "Shield 2 must abort and rollback on IMU slip mismatch");
        assert!(d_m > 3.0);
        assert_eq!(state, init);
    }

    #[test]
    fn test_shield3_hft_cbf_gateway_clips_unsafe_cutin() {
        let gw = CbfPreTradeGateway {
            headway_time: 1.5,
            alpha: 2.0,
            friction_mu: 0.85,
            g: 9.81,
        };
        // Neural policy commands +3.0 m/s^2 acceleration + 9.0 m/s^2 hard swerve during close cut-in
        let u_nom = ControlCommand { a_x: 3.0, a_y: 9.0 };
        let (u_safe, intervened, _h) = gw.filter_control(u_nom, 12.0, 25.0, 15.0);
        assert!(intervened, "Shield 3 must clip unsafe acceleration and friction breach");
        assert!(u_safe.a_x < 0.0, "Must command braking when headway is violated");
        let total_acc = (u_safe.a_x * u_safe.a_x + u_safe.a_y * u_safe.a_y).sqrt();
        assert!(total_acc <= 0.85 * 9.81 + 1e-5, "Must stay inside Kamm friction circle");
    }
}
