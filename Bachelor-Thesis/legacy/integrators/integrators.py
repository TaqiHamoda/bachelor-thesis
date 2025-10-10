import numpy as np

from utils.optimization import Function


class Integrator:
    def __init__(self, h: float) -> None:
        self.h = h
        self.ys: list[np.ndarray] = []


    def _integrate(self, f: Function, t: float, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


    def integrate(self, f: Function, y0: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
        if t_end - t_start < self.h:
            return y0

        y = y0
        self.ys = [y]

        t = t_start
        while t < t_end:
            y = self._integrate(f, t, y)
            t += self.h

            self.ys.append(y)

        return y


class ForwardEuler(Integrator):
    def _integrate(self, f: Function, t: float, y: np.ndarray) -> np.ndarray:
        return y + self.h * f.derive(t, y)
