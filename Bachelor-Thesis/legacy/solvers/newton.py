import numpy as np

from utils.optimization import Function, approx_fprime
from solvers.solvers import Solver, SolverMomentum, SolverNesterov, SolverAdam, SolverAdamW


# Newton Methods
class Newton(Solver):
    def _get_dx(self, f: Function, x: np.ndarray, f_x: np.ndarray) -> np.ndarray:
        jac = approx_fprime(f, x, f_x)
        grad = np.linalg.solve(jac, f_x)

        return grad


class NewtonMomentum(Newton, SolverMomentum):
    pass


class NewtonNesterov(Newton, SolverNesterov):
    pass


class NewtonAdam(Newton, SolverAdam):
    pass


class NewtonAdamW(Newton, SolverAdamW):
    pass