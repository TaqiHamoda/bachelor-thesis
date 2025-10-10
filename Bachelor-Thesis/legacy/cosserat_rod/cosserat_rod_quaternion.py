import numpy as np
from integrators.integrators import Integrator
from solvers.solvers import Solver
from utils.tube import Tube
from utils.optimization import quaternion_to_rotation

from cosserat_rod.cosserat_rod import CosseratRod


class CosseratRodQuaternion(CosseratRod):
    def __init__(self, ctcr: list[Tube], joints: np.ndarray, solver: Solver, integrator: Integrator) -> None:
        super().__init__(ctcr, joints, solver, integrator)

        self.state_idx = 7


    def _construct_p_rot(self, psi_0: float) -> np.ndarray:
        p_init = np.zeros((3, ))
        q_init = np.array([np.cos(0.5*psi_0), 0, 0, np.sin(0.5*psi_0)])

        return np.concatenate((p_init, q_init))


    def _construct_p_rot_dot(self, y: np.ndarray, u0: np.ndarray) -> np.ndarray:
        p_dot = np.array([2*(y[4]*y[6] + y[3]*y[5]), 2*(y[5]*y[6] - y[3]*y[4]), y[3]*y[3] - y[4]*y[4] - y[5]*y[5] + y[6]*y[6]])

        Wt = np.array([
            [-y[4], -y[5], -y[6]],
            [ y[3], -y[6],  y[5]],
            [ y[6],  y[3], -y[4]],
            [-y[5],  y[4],  y[3]]
        ])
        q_dot = Wt@u0/2

        return np.concatenate((p_dot, q_dot))


    def construct_tip(self) -> np.ndarray:
        tip = np.zeros((4, 4))

        # Construct Rotation
        tip[:3, :3] = quaternion_to_rotation(self.y[3:self.state_idx])

        # Construct Position
        tip[:3, 3] = self.y[:3]
        tip[3, 3] = 1

        return tip


