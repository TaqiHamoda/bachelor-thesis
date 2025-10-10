import numpy as np

from utils.optimization import Function
from integrators.integrators import Integrator

# Reference https://en.wikipedia.org/wiki/Linear_multistep_method
class AdamsBashforth(Integrator):
    A = (
        (3/2, -1/2),
        (23/12, -16/12, 5/12),
        (55/24, -59/24, 37/24, -9/24)
    )

    def __init__(self, h: float) -> None:
        super().__init__(h)
        self.f_t = [None, None, None, None]


    def integrate(self, f: Function, y0: np.ndarray, t_start: float, t_end: float) -> list[np.ndarray]:
        self.f_t = [None, None, None, None]
        return super().integrate(f, y0, t_start, t_end)


class AB2(AdamsBashforth):
    def _integrate(self, f: Function, t: float, y: np.ndarray) -> np.ndarray:
        if self.f_t[0] is None:
            self.f_t[0] = f.derive(t, y)
            return y + self.h*self.f_t[0]

        self.f_t[1] = self.f_t[0]
        self.f_t[0] = f.derive(t, y)

        return y + self.h*(self.A[0][0]*self.f_t[0] + self.A[0][1]*self.f_t[1])


class AB3(AB2):
    def _integrate(self, f: Function, t: float, y: np.ndarray) -> np.ndarray:
        if self.f_t[0] is None or self.f_t[1] is None:
            return super()._integrate(f, t, y)

        self.f_t[2] = self.f_t[1]
        self.f_t[1] = self.f_t[0]
        self.f_t[0] = f.derive(t, y)

        return y + self.h*(self.A[1][0]*self.f_t[0] + self.A[1][1]*self.f_t[1] + self.A[1][2]*self.f_t[2])


class AB4(AB3):
    def _integrate(self, f: Function, t: float, y: np.ndarray) -> np.ndarray:
        if self.f_t[0] is None or self.f_t[1] is None or self.f_t[2] is None:
            return super()._integrate(f, t, y)

        self.f_t[3] = self.f_t[2]
        self.f_t[2] = self.f_t[1]
        self.f_t[1] = self.f_t[0]
        self.f_t[0] = f.derive(t, y)

        return y + self.h*(self.A[2][0]*self.f_t[0] + self.A[2][1]*self.f_t[1] + self.A[2][2]*self.f_t[2] + self.A[2][3]*self.f_t[3])
