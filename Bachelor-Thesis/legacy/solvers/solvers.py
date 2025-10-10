import numpy as np
import warnings

from utils.optimization import Function, EPSILON

MAX_ITERATION = 100
TOLERANCE = 1e-06  # Based on scipy default solver tolerance


# Solver Class
class Solver:
    def __init__(self) -> None:
        self.err_list = []


    def _get_dx(self, f: Function, x: np.ndarray, f_x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


    def get_dx(self, f: Function, x: np.ndarray, f_x: np.ndarray) -> np.ndarray:
        return self._get_dx(f, x, f_x)


    def solve(self, f: Function, x: np.ndarray) -> np.ndarray:
        self.err_list = []

        f_x = f.evaluate(x)
        if np.linalg.norm(f_x) < TOLERANCE:
            return x

        for _ in range(MAX_ITERATION):
            dx = self.get_dx(f, x, f_x)
            x = x - dx

            f_x = f.evaluate(x)

            err = np.linalg.norm(f_x)
            self.err_list.append(err)
            if err < TOLERANCE:
                return x

        warnings.warn(f"Solver didn't converge after {MAX_ITERATION} iterations", RuntimeWarning)

        return x


class SolverMomentum(Solver):
    def __init__(self) -> None:
        super().__init__()
        self.moment = np.zeros((0, ))
        self.momentum = 0

        self._decay_rate = 0.9


    def get_dx(self, f: Function, x: np.ndarray, f_x: np.ndarray) -> np.ndarray:
        # Reference: https://www.deeplearningbook.org/contents/optimization.html
        self.moment = (1 - self.momentum)*super().get_dx(f, x, f_x) + self.momentum*self.moment
        self.momentum *= 0.9

        return self.moment


    def solve(self, f: Function, x: np.ndarray) -> np.ndarray:
        self.moment = np.zeros(x.shape)

        return super().solve(f, x)


class SolverNesterov(SolverMomentum):
    def get_dx(self, f: Function, x: np.ndarray, f_x: np.ndarray) -> np.ndarray:
        # Reference: https://www.deeplearningbook.org/contents/optimization.html
        return super().get_dx(f, x - self.momentum*self.moment, f_x)


class SolverAdam(Solver):
    def __init__(self) -> None:
        super().__init__()
        self.betas = [0.9, 0.999]
        self.moment = np.zeros((0, ))
        self.acceleration = np.zeros((0, ))

        self._decay_rate = 0.9


    def get_dx(self, f: Function, x: np.ndarray, f_x: np.ndarray) -> np.ndarray:
        # Reference: https://arxiv.org/pdf/1904.09237.pdf
        # Reference: https://arxiv.org/pdf/2208.09632.pdf  (Why only beta[0] is being dampened)

        dx = super().get_dx(f, x, f_x)

        alpha = np.sqrt(1 - self.betas[1]) / (1 - self.betas[0])
        self.moment = (1 - self.betas[0])*dx + self.betas[0]*self.moment
        self.acceleration = (1 - self.betas[1])*np.power(dx, 2) + self.betas[1]*self.acceleration

        dx = alpha*self.moment / (np.sqrt(self.acceleration) + EPSILON)

        self.betas[0] *= self._decay_rate

        return dx


    def solve(self, f: Function, x: np.ndarray) -> np.ndarray:
        self.moment = np.zeros(x.shape)
        self.acceleration = np.zeros(x.shape)

        return super().solve(f, x)


class SolverAdamW(SolverAdam):
    def get_dx(self, f: Function, x: np.ndarray, f_x: np.ndarray) -> np.ndarray:
        # Reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        super().get_dx(f, x, f_x)  # Update betas, moment and acceleration

        m_hat = self.moment/(1 - self.betas[0])
        a_hat = self.acceleration/(1 - self.betas[1])

        dx = m_hat / (np.sqrt(a_hat) + EPSILON)

        return dx