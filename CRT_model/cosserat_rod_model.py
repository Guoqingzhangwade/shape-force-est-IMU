"""Cosserat-rod statics model for a **single-segment** TDCR.
   • Supports arbitrary tendon count (default = 4).
   • Solves force & moment equilibrium via shooting + SciPy.
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import solve, svd, norm
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from typing import Sequence, Union
from utils import hat, unit, block33_to_mat

E3 = np.array([0.0, 0.0, 1.0])  # backbone tangential basis

class CosseratRodModel:
    """Static Cosserat rod for tendon-driven CR (single segment)."""

    def __init__(self,
                 length: float = 0.1,
                 backbone_radius: float = 5e-4,
                 youngs_modulus: float = 60e9,
                 tendon_routing: Sequence[np.ndarray] | None = None,
                 tendon_count: int = 4,
                 num_disks: int = 20):
        self.L = length
        self.Ro = backbone_radius
        self.E = youngs_modulus
        self.G = self.E / (2 * (1 + 0.3))  # assume ν≈0.3
        self.I = np.pi * self.Ro ** 4 / 4
        self.A = np.pi * self.Ro ** 2
        # stiffness (diag) matrices (isotropic circular tube)
        self.Kse = np.diag([self.G * self.A,
                            self.G * self.A,
                            self.E * self.A])
        self.Kbt = np.diag([self.E * self.I,
                            self.E * self.I,
                            2 * self.G * self.I])
        # Tendon routing
        if tendon_routing is None:
            pr = 0.008  # default pitch radius
            angles = np.linspace(0, 2*np.pi, tendon_count, endpoint=False)
            tendon_routing = [np.array([pr*np.cos(a), pr*np.sin(a), 0.0])
                              for a in angles]
        self.r = list(tendon_routing)
        self.n_tendon = len(self.r)
        self.N = num_disks  # discrete integration steps

    # ------------------------------------------------------------------
    # ODE right‑hand side  y' = f(s, y)
    # y = [p(3)  R(9)  v(3)  u(3)]  (total 18)
    # ------------------------------------------------------------------
    def _ode(self, s: float, y: np.ndarray, tau: np.ndarray) -> np.ndarray:
        p   = y[ 0:3]
        R   = y[ 3:12].reshape(3,3)
        v   = y[12:15]
        u   = y[15:18]
        u_hat = hat(u)
        v_hat = hat(v)

        # Pre‑compute routing skew matrices
        r_hat = [hat(ri) for ri in self.r]

        # Tendon unit directions pb_dot_i in body frame
        pb_dot = [u_hat @ ri + v for ri in self.r]

        # A_i matrices Eq.(15) Rao2021
        A = [-tau_i / norm(pb_dot_i)**3 * hat(pb_dot_i) @ hat(pb_dot_i)
             for tau_i, pb_dot_i in zip(tau, pb_dot)]
        A_total = sum(A)

        # B_i = r_hat_i * A_i
        B = [rhi @ Ai for rhi, Ai in zip(r_hat, A)]
        B_total = sum(B)

        # G & H matrices
        G = sum([-Ai @ rhi for Ai, rhi in zip(A, r_hat)])
        H = sum([-Bi @ rhi for Bi, rhi in zip(B, r_hat)])

        # a_i & b_i vectors
        a = [Ai @ u_hat @ pb_dot_i for Ai, pb_dot_i in zip(A, pb_dot)]
        a_total = sum(a)
        b = [rhi @ ai for rhi, ai in zip(r_hat, a)]
        b_total = sum(b)

        # Elastic load vectors (assuming straight reference)
        c = (-u_hat @ self.Kbt @ u
             - v_hat @ self.Kse @ (v - E3)
             - R.T @ np.zeros(3)  # no distributed ext mom
             - b_total)
        d = (-u_hat @ self.Kse @ (v - E3)
             - R.T @ np.zeros(3)  # no distributed ext force
             - a_total)
        rhs6 = np.hstack([d, c])  # (6,)

        # Build 6×6 block matrix
        upper = np.hstack([self.Kse + A_total, G])
        lower = np.hstack([B_total, self.Kbt + H])
        M66 = np.vstack([upper, lower])

        vu_dot = solve(M66, rhs6)  # 6‑vector [v_dot, u_dot]
        v_dot = vu_dot[:3]
        u_dot = vu_dot[3:]

        p_dot = R @ v
        R_dot = R @ u_hat

        # Flatten derivatives to 18‑vector
        dy = np.empty_like(y)
        dy[ 0:3]  = p_dot
        dy[ 3:12] = R_dot.reshape(-1)
        dy[12:15] = v_dot
        dy[15:18] = u_dot
        return dy

    # ------------------------------------------------------------------
    # Boundary‑value shooting: solve initial (v0,u0) s.t. tip equilibrium
    # ------------------------------------------------------------------
    def forward_kinematics(self,
                           tau: np.ndarray,
                           f_ext: np.ndarray | None = None,
                           l_ext: np.ndarray | None = None,
                           guess: np.ndarray | None = None,
                           return_states: bool = False,
                           s_eval: Union[None, Sequence[float]] = None):
        """Solve static equilibrium for given tendon *tensions* τ (N).
        Returns tip frame T ∈ SE(3).  If *return_states* is True the
        complete discretised backbone states are also returned.
        """
        tau = np.asarray(tau, dtype=float).ravel()
        assert tau.size == self.n_tendon, "tau length ≠ n_tendon"
        f_ext = np.zeros(3) if f_ext is None else np.asarray(f_ext)
        l_ext = np.zeros(3) if l_ext is None else np.asarray(l_ext)

        def shoot(x):
            """Residual of tip equilibrium for optimiser."""
            v0, u0 = x[:3], x[3:]
            y0 = np.zeros(18)
            y0[0:3] = 0.0  # p0
            y0[3:12] = np.eye(3).reshape(-1)  # R0
            y0[12:15] = v0
            y0[15:18] = u0
            sol = solve_ivp(self._ode, (0, self.L), y0,
                            t_eval=s_eval,
                            args=(tau,), max_step=self.L/self.N,
                            rtol=1e-6, atol=1e-8)
            yL = sol.y[:,-1] # state at s=L
            R = yL[3:12].reshape(3,3)
            vL = yL[12:15]
            uL = yL[15:18]
            # internal loads at tip
            n_L = R @ self.Kse @ (vL - E3)
            m_L = R @ self.Kbt @ uL
            # tendon resultant loads at tip
            u_hat = hat(uL)
            pbdot = [u_hat @ ri + vL for ri in self.r]
            F = sum([-ti/ norm(pb) * (R @ pb) for ti,pb in zip(tau,pbdot)])
            Lmom = sum([-ti/ norm(pb) * np.cross(R @ ri, R @ pb)
                        for ti,ri,pb in zip(tau,self.r,pbdot)])
            # equilibrium residuals
            res = np.hstack([n_L - F - f_ext,
                              m_L - Lmom - l_ext])
            return res

        if guess is None:
            guess = np.array([0,0,1, 0,0,0], dtype=float)  # straight
        sol = least_squares(shoot, guess, xtol=1e-10, ftol=1e-10,
                             gtol=1e-10, method='lm', max_nfev=500)
        if not sol.success or norm(sol.fun) > 1e-6:
            raise RuntimeError(f"Shooting failed: {sol.message}, ‖res‖={norm(sol.fun):.2e}")

        # with optimal init, integrate once more & return trajectory
        v0_opt, u0_opt = sol.x[:3], sol.x[3:]
        y0 = np.zeros(18)
        y0[3:12] = np.eye(3).reshape(-1)
        y0[12:15] = v0_opt; y0[15:18] = u0_opt
        traj = solve_ivp(self._ode, (0, self.L), y0, args=(tau,),
                         t_eval=s_eval,
                         max_step=self.L/self.N, rtol=1e-6, atol=1e-8)

        yL = traj.y[:,-1]
        T_tip = np.eye(4)
        T_tip[0:3,0:3] = yL[3:12].reshape(3,3)
        T_tip[0:3,3]   = yL[0:3]

        if return_states:
            return T_tip, traj
        return T_tip
