import numpy as np

from utils.optimization import Function
from integrators.integrators import Integrator

# Reference: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
# Note: For some reason, rk2 and rk3 perform better than rk4 in the stiff case
# All the methods listed on wikipedia were tested and the best performing were picked

class RK2(Integrator):
    def _integrate(self, f: Function, t: float, y: np.ndarray) -> np.ndarray:
        """
        Based on Ralston's method
        """
        a21 = 2/3

        c2 = 2/3

        b1 = 1/4
        b2 = 3/4

        Y1 = y

        f_t_1 = f.derive(t, Y1)
        Y2 = Y1 + self.h*a21*f_t_1

        return Y1 + self.h*(b1*f_t_1 + b2*f.derive(t + c2*self.h, Y2))


class RK3(Integrator):
    def _integrate(self, f: Function, t: float, y: np.ndarray) -> np.ndarray:
        """
        Based on Ralston's third-order method
        """
        a21 = 1/2

        a31 = 0
        a32 = 3/4

        b1 = 2/9
        b2 = 1/3
        b3 = 4/9

        c2 = 1/2
        c3 = 3/4

        Y1 = y

        f_t_1 = f.derive(t, Y1)
        Y2 = Y1 + self.h*a21*f_t_1

        f_t_2 = f.derive(t + c2*self.h, Y2)
        Y3 = Y1 + self.h*(a31*f_t_1 + a32*f_t_2)

        return Y1 + self.h*(b1*f_t_1 + b2*f_t_2 + b3*f.derive(t + c3*self.h, Y3))


class RK4(Integrator):
    def _integrate(self, f: Function, t: float, y: np.ndarray) -> np.ndarray:
        """
        Based on 3/8-rule fourth-order method
        """
        a21 = 1/2

        a31 = 0
        a32 = 1/2

        a41 = 0
        a42 = 0
        a43 = 1

        b1 = 1/6
        b2 = 1/3
        b3 = 1/3
        b4 = 1/6

        c2 = 1/2
        c3 = 1/2
        c4 = 1

        Y1 = y
        
        f_t_1 = f.derive(t, Y1)
        Y2 = Y1 + self.h*a21*f_t_1

        f_t_2 = f.derive(t + c2*self.h, Y2)
        Y3 = Y1 + self.h*(a31*f_t_1 + a32*f_t_2)

        f_t_3 = f.derive(t + c3*self.h, Y3)
        Y4 = Y1 + self.h*(a41*f_t_1 + a42*f_t_2 + a43*f_t_3)

        return Y1 + self.h*(b1*f_t_1 + b2*f_t_2 + b3*f_t_3 + b4*f.derive(t + c4*self.h, Y4))
