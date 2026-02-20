from __future__ import annotations
import numpy as np
from numpy.linalg import solve, norm
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from typing import Sequence, Union
from utils import hat

E3 = np.array([0.0, 0.0, 1.0])  # backbone tangential basis (body frame)

class CosseratRodModel:
    """Static Cosserat rod for tendon-driven CR (single segment).
    - extensible/shearable: y = [p(3), R(9), v(3), u(3)]
    - inextensible (Kirchhoff): y = [p(3), R(9), n(3), u(3)]
    """

    def __init__(self,
                 length: float = 0.1,
                 backbone_radius: float = 5e-4,
                 youngs_modulus: float = 60e9,
                 tendon_routing: Sequence[np.ndarray] | None = None,
                 tendon_count: int = 4,
                 num_disks: int = 20,
                 inextensible: bool = False):
        self.L = length
        self.Ro = backbone_radius
        self.E = youngs_modulus
        self.G = self.E / (2 * (1 + 0.3))  # assume nu≈0.3
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
            pr = 0.008
            angles = np.linspace(0, 2*np.pi, tendon_count, endpoint=False)
            tendon_routing = [np.array([pr*np.cos(a), pr*np.sin(a), 0.0])
                              for a in angles]
        self.r = list(tendon_routing)
        self.n_tendon = len(self.r)

        self.N = num_disks
        self.inextensible = bool(inextensible)

        # Precompute constant hats
        self._r_hat = [hat(ri) for ri in self.r]
        self._v_fixed = E3.copy()
        self._v_hat_fixed = hat(E3)

    # ------------------------------------------------------------------
    # ODE right-hand side y' = f(s, y)
    # ------------------------------------------------------------------
    def _ode(self, s: float, y: np.ndarray, tau: np.ndarray) -> np.ndarray:
        # Common unpack
        p = y[0:3]
        R = y[3:12].reshape(3, 3)

        if self.inextensible:
            # Kirchhoff rod: y = [p, R, n, u]
            n = y[12:15]          # spatial internal force
            u = y[15:18]          # body curvature
            v = self._v_fixed
            v_hat = self._v_hat_fixed
        else:
            # General Cosserat: y = [p, R, v, u]
            v = y[12:15]          # body strain (shear/extension)
            u = y[15:18]          # body curvature
            v_hat = hat(v)

        u_hat = hat(u)

        # Tendon unit directions pb_dot_i in body frame: p'_b = u×r + v
        pb_dot = [u_hat @ ri + v for ri in self.r]

        # A_i matrices (Rucker 2011 eq.(15)-style form)
        # NOTE: keep your original expression; assumes tau_i >= 0
        A_list = []
        for tau_i, pb_i in zip(tau, pb_dot):
            pb_norm = norm(pb_i)
            # safeguard (rare)
            pb_norm = max(pb_norm, 1e-12)
            A_i = -tau_i / (pb_norm**3) * (hat(pb_i) @ hat(pb_i))
            A_list.append(A_i)
        A_total = sum(A_list)

        # B_i = r_hat_i * A_i
        B_list = [rhi @ Ai for rhi, Ai in zip(self._r_hat, A_list)]
        B_total = sum(B_list)

        # G & H matrices
        G = sum([-Ai @ rhi for Ai, rhi in zip(A_list, self._r_hat)])
        H = sum([-Bi @ rhi for Bi, rhi in zip(B_list, self._r_hat)])

        # a_i & b_i vectors
        a_list = [Ai @ u_hat @ pb_i for Ai, pb_i in zip(A_list, pb_dot)]
        a_total = sum(a_list)
        b_list = [rhi @ ai for rhi, ai in zip(self._r_hat, a_list)]
        b_total = sum(b_list)

        # External distributed loads (set to zero here; you can extend later)
        f_e = np.zeros(3)  # spatial distributed force
        l_e = np.zeros(3)  # spatial distributed moment

        # Kinematics
        p_dot = R @ v
        R_dot = R @ u_hat

        dy = np.empty_like(y)
        dy[0:3] = p_dot
        dy[3:12] = R_dot.reshape(-1)

        if self.inextensible:
            # ----------------------------
            # Kirchhoff / inextensible case
            # ----------------------------
            # c = - u^ Kbt u  - v^ R^T n  - R^T l_e  - b
            c = (-u_hat @ self.Kbt @ u
                 -v_hat @ (R.T @ n)
                 -R.T @ l_e
                 -b_total)

            # u' = (Kbt + H)^(-1) c
            u_dot = solve(self.Kbt + H, c)

            # n' = -f_e - R( a + G u' )   (since v' = 0)
            n_dot = -f_e - R @ (a_total + G @ u_dot)

            dy[12:15] = n_dot
            dy[15:18] = u_dot
        else:
            # ----------------------------
            # General shear/extension case (your original block solve)
            # ----------------------------
            c = (-u_hat @ self.Kbt @ u
                 -v_hat @ self.Kse @ (v - E3)
                 -R.T @ l_e
                 -b_total)
            d = (-u_hat @ self.Kse @ (v - E3)
                 -R.T @ f_e
                 -a_total)
            rhs6 = np.hstack([d, c])

            upper = np.hstack([self.Kse + A_total, G])
            lower = np.hstack([B_total, self.Kbt + H])
            M66 = np.vstack([upper, lower])

            vu_dot = solve(M66, rhs6)
            v_dot = vu_dot[:3]
            u_dot = vu_dot[3:]

            dy[12:15] = v_dot
            dy[15:18] = u_dot

        return dy

    # ------------------------------------------------------------------
    # Boundary-value shooting
    # ------------------------------------------------------------------
    def forward_kinematics(self,
                           tau: np.ndarray,
                           f_ext: np.ndarray | None = None,
                           l_ext: np.ndarray | None = None,
                           guess: np.ndarray | None = None,
                           return_states: bool = False,
                           s_eval: Union[None, Sequence[float]] = None,
                           solver_opts: dict | None = None):
        tau = np.asarray(tau, dtype=float).ravel()
        assert tau.size == self.n_tendon, "tau length ≠ n_tendon"
        f_ext = np.zeros(3) if f_ext is None else np.asarray(f_ext, dtype=float)
        l_ext = np.zeros(3) if l_ext is None else np.asarray(l_ext, dtype=float)

        def shoot(x):
            """Residual of tip equilibrium for optimiser."""
            y0 = np.zeros(18)
            y0[0:3] = 0.0
            y0[3:12] = np.eye(3).reshape(-1)

            if self.inextensible:
                # x = [n0(3), u0(3)]
                n0, u0 = x[:3], x[3:]
                y0[12:15] = n0
                y0[15:18] = u0
            else:
                # x = [v0(3), u0(3)]
                v0, u0 = x[:3], x[3:]
                y0[12:15] = v0
                y0[15:18] = u0

            sol = solve_ivp(self._ode, (0, self.L), y0,
                            t_eval=s_eval, args=(tau,),
                            max_step=self.L / self.N,
                            rtol=1e-6, atol=1e-8)

            yL = sol.y[:, -1]
            R_L = yL[3:12].reshape(3, 3)
            u_L = yL[15:18]
            u_hat = hat(u_L)

            # Internal loads at tip
            if self.inextensible:
                n_L = yL[12:15]                # spatial internal force (state)
            else:
                v_L = yL[12:15]
                n_L = R_L @ self.Kse @ (v_L - E3)

            m_L = R_L @ self.Kbt @ u_L

            # Tendon resultant loads at tip (same expression; v = E3 if inextensible)
            v_use = E3 if self.inextensible else yL[12:15]
            pb_dot = [u_hat @ ri + v_use for ri in self.r]

            # Spatial tendon force & moment resultants
            F = sum([-ti / max(norm(pb), 1e-12) * (R_L @ pb)
                     for ti, pb in zip(tau, pb_dot)])
            Lmom = sum([-ti / max(norm(pb), 1e-12) * np.cross(R_L @ ri, R_L @ pb)
                        for ti, ri, pb in zip(tau, self.r, pb_dot)])

            res = np.hstack([n_L - F - f_ext,
                             m_L - Lmom - l_ext])
            return res

        # Default guess
        if guess is None:
            if self.inextensible:
                guess = np.zeros(6)  # [n0,u0]
            else:
                guess = np.array([0, 0, 1, 0, 0, 0], dtype=float)  # [v0,u0]

        solver_opts = solver_opts or {}
        sol = least_squares(
            shoot, guess,
            xtol=solver_opts.get("xtol", 1e-10),
            ftol=solver_opts.get("ftol", 1e-10),
            gtol=solver_opts.get("gtol", 1e-10),
            method=solver_opts.get("method", "lm"),
            max_nfev=solver_opts.get("max_nfev", 500),
        )

        if (not sol.success) or (norm(sol.fun) > 1e-6):
            raise RuntimeError(f"Shooting failed: {sol.message}, ‖res‖={norm(sol.fun):.2e}")

        # Integrate again with optimal init
        y0 = np.zeros(18)
        y0[3:12] = np.eye(3).reshape(-1)

        if self.inextensible:
            n0_opt, u0_opt = sol.x[:3], sol.x[3:]
            y0[12:15] = n0_opt
            y0[15:18] = u0_opt
        else:
            v0_opt, u0_opt = sol.x[:3], sol.x[3:]
            y0[12:15] = v0_opt
            y0[15:18] = u0_opt

        traj = solve_ivp(self._ode, (0, self.L), y0,
                         args=(tau,), t_eval=s_eval,
                         max_step=self.L / self.N,
                         rtol=1e-6, atol=1e-8)

        yL = traj.y[:, -1]
        T_tip = np.eye(4)
        T_tip[0:3, 0:3] = yL[3:12].reshape(3, 3)
        T_tip[0:3, 3] = yL[0:3]

        if return_states:
            return T_tip, traj
        return T_tip
