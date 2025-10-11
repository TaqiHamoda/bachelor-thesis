import numpy as np

from utils.optimization import Function, approx_fprime, EPSILON
from solvers.solvers import Solver, SolverMomentum, SolverNesterov, SolverAdam, SolverAdamW
from solvers.solvers import TOLERANCE, MAX_ITERATION


# Quasi-Newton (Secant) Methods: Broyden
class Broyden(Solver):
    def __init__(self, reuse_jac: bool = False, jac_step: float = 0) -> None:
        super().__init__()
        self.grad = np.zeros((0, ))
        self.jac_inv = np.zeros((0, ))
        self._initialized = False

        self.reuse_jac = reuse_jac
        self.jac_step = jac_step


    def _get_dx(self, f: Function, x: np.ndarray, f_x: np.ndarray) -> np.ndarray:
        """
        Based on Broyden's good method (gradient update)

        Reference: https://faculty.math.illinois.edu/~mlavrov/docs/484-spring-2019/ch3lec6.pdf
        """
        if not self._initialized:
            self._initialized = True

            # If asked to use large step size
            step_size = f.integrator.h
            if self.jac_step > 0:
                f.integrator.h = self.jac_step
                f_x = f.evaluate(x)

            # Do one iteration of Newton
            self.jac_inv = np.linalg.inv(approx_fprime(f, x, f_x))  # Only solve for the inverse jacobian once
            self.grad = self.jac_inv@f_x

            # Reset to old step size
            f.integrator.h = step_size

            return self.grad

        g = -self.grad
        g_next = self.jac_inv@f_x

        self.jac_inv -= np.outer(g_next, g@self.jac_inv) / (g@(g + g_next) + EPSILON)
        self.grad = self.jac_inv@f_x

        return self.grad


    def solve(self, f: Function, x: np.ndarray) -> np.ndarray:
        if not self.reuse_jac:
            self._initialized = False

        return super().solve(f, x)


class BroydenMomentum(Broyden, SolverMomentum):
    pass


class BroydenNesterov(Broyden, SolverNesterov):
    pass


class BroydenAdam(Broyden, SolverAdam):
    pass


class BroydenAdamW(Broyden, SolverAdamW):
    pass