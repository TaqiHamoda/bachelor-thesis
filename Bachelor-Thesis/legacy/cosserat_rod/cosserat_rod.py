import numpy as np

from utils.tube import Tube
from utils.optimization import Function
from solvers.solvers import Solver
from integrators.integrators import Integrator


class CosseratRod(Function):
    """
    Cosserat Rod Modelling Algorithm for a 3-segment CTCR
    """
    n = 3

    def __init__(self, ctcr: list[Tube], joints: np.ndarray, solver: Solver, integrator: Integrator) -> None:
        if len(ctcr) != self.n:
            raise Exception(f"CTCR must have {self.n} segments.")

        self.idx = 0
        self._shooting_iter = 0  # Counter for how many times derive has been called

        self.y = np.zeros((0, ))

        self.KBT = np.array([c.kbt for c in ctcr])
        self.U = np.array([c.u for c in ctcr])

        self.ALPHAS = joints[:, 0]
        self.BETAS = joints[:, 1]

        self.L = np.array([c.L for c in ctcr])
        self.Ls = np.array([c.Ls for c in ctcr])

        self.ctcr = ctcr

        self._initialize()

        self.state_idx = 12

        self.solver = solver
        self.integrator = integrator

        self.backbone = []


    def _initialize(self):
        self.tube_ends = [self.ctcr[i].L + self.BETAS[i] for i in range(self.n)]

        self.tube_lengths = [0,] + self.tube_ends +\
            [self.ctcr[i].Ls + self.BETAS[i] for i in range(self.n)]
        self.tube_lengths = sorted(self.tube_lengths)

        self.idx = self.tube_lengths.index(0)


    def _construct_p_rot(self, psi_0: float) -> np.ndarray:
        c_psi = np.cos(psi_0)
        s_psi = np.sin(psi_0)

        p_init = np.zeros((3, ))

        R_init = np.array([
            [c_psi, -s_psi, 0],
            [s_psi,  c_psi, 0],
            [    0,      0, 1]
        ]).reshape((9,))

        return np.concatenate((p_init, R_init))


    def _construct_p_rot_dot(self, y: np.ndarray, u0: np.ndarray) -> np.ndarray:
        R = np.reshape(y[3:12], (3, 3))

        p = R[:, 2]

        u_hat = np.array([
            [     0, -u0[2],  u0[1]],
            [ u0[2],      0, -u0[0]],
            [-u0[1],  u0[0],      0]
        ])
        R_dot = R@u_hat

        return np.concatenate((p, R_dot.reshape((9,))))


    def _is_tube(self, i: int) -> bool:
        return (self.tube_lengths[self.idx] + self.tube_lengths[self.idx + 1])/2 < self.BETAS[i] + self.L[i]


    def _is_straight(self, i: int) -> bool:
        return (self.tube_lengths[self.idx] + self.tube_lengths[self.idx + 1])/2 < self.BETAS[i] + self.Ls[i]


    def construct_tip(self) -> np.ndarray:
        tip = np.zeros((4, 4))

        # Construct Rotation (Orientation)
        tip[:3, :3] = self.y[3:12].reshape((3, 3))

        # Construct Position
        tip[:3, 3] = self.y[:3]
        tip[3, 3] = 1

        return tip


    def derive(self, t: float, y: np.ndarray) -> np.ndarray:
        # Based on: https://ieeexplore.ieee.org/document/5559519
        # https://ieeexplore.ieee.org/document/5980351

        self._shooting_iter += 1

        uz = y[self.state_idx:self.state_idx + self.n]
        psi = y[self.state_idx + self.n:self.state_idx + 2*self.n]

        theta = psi[1:] - psi[0]
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)

        Rtheta = np.zeros((self.n - 1, 3, 3))
        Rtheta[:, 0, 0] = c_theta
        Rtheta[:, 0, 1] = -s_theta
        Rtheta[:, 1, 0] = s_theta
        Rtheta[:, 1, 1] = c_theta
        Rtheta[:, 2, 2] = 1

        e3 = np.array([0, 0, 1])
        u = np.zeros((self.n, 3))

        # Get tube info
        u_star = np.zeros((self.n, 3))

        if not self._is_straight(0):
            u_star[0, :] = self.ctcr[0].u

        k = np.zeros((3, 3))

        ks = np.zeros((self.n, 3, 3))
        ks[0, :, :] = self.ctcr[0].kbt
        k += ks[0]

        u[0] = ks[0]@u_star[0]
        for i in range(1, self.n):
            if self._is_tube(i) and not self._is_straight(i):
                u_star[i, :] = self.ctcr[i].u

            if self._is_tube(i):
                ks[i, :, :] = self.ctcr[i].kbt

            k += ks[i]

            u[0] += Rtheta[i - 1]@ks[i]@u_star[i] - ks[i][2, 2]*(uz[i] - uz[0])*e3

        invk = np.diag(1/np.diag(k))
        u[0] = invk@u[0]

        uz_dot = np.zeros((self.n,))
        uz_dot[0] = ks[0][0, 0]/ks[0][2, 2]*(u[0][0]*u_star[0][1] - u[0][1]*u_star[0][0])

        for i in range(1, self.n):
            u[i] = Rtheta[i - 1].T@u[0] + (uz[i] - uz[0])*e3

            if (self.tube_lengths[self.idx] + self.tube_lengths[self.idx + 1])/2 < self.BETAS[i] + self.ctcr[i].L:
                uz_dot[i] = ks[i][0, 0]/ks[i][2, 2]*(u[i][0]*u_star[i][1] - u[i][1]*u_star[i][0])

        p_rot_dot = self._construct_p_rot_dot(y, u[0])

        return np.concatenate((p_rot_dot, uz_dot, uz, (1, )))


    def evaluate(self, x: np.ndarray) -> np.ndarray:
        # Based on: https://ieeexplore.ieee.org/document/5980351
        self.backbone = [[]]

        psi_init = self.ALPHAS + x*np.abs(self.BETAS)
        p_rot_init = self._construct_p_rot(psi_init[0])

        self.y = np.concatenate((p_rot_init, x, psi_init, (0,)))

        start_idx = self.idx
        while self.idx < 2*self.n:
            if self.tube_lengths[self.idx] in self.tube_ends:
                self.backbone.append([])

            self.y = self.integrator.integrate(self, self.y, self.tube_lengths[self.idx], self.tube_lengths[self.idx + 1])
            self.backbone[-1].extend(np.array(self.integrator.ys)[:, :self.state_idx].tolist())  # Add the position and orientation of backbone

            self.idx += 1
        self.idx = start_idx

        return self.y[self.state_idx:self.state_idx + self.n]


    def solve(self, u_init_guess: np.ndarray) -> np.ndarray:
        self._shooting_iter = 0
        return self.solver.solve(self, u_init_guess)
