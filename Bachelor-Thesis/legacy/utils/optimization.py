import numpy as np

EPSILON = 1.4901161193847656e-08  # Based on scipy default epsilon


class Function:
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def derive(self, t: float, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    

# Utils
def approx_fprime(f: Function, x: np.ndarray, f_x: np.ndarray, eps=EPSILON) -> np.ndarray:
    jac = np.zeros((x.size, x.size))

    e_i = np.zeros((x.size,))
    for i in range(x.size):
        e_i[i] = eps
        df = (f.evaluate(x + e_i) - f_x)/eps
        jac[:, i] = df

        e_i[i] = 0

    return jac


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    # Reference: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    q_r, q_i, q_j, q_k = q

    trans = np.array([
        [1 - 2*(q_j**2 + q_k**2),   2*(q_i*q_j - q_k*q_r),   2*(q_i*q_k + q_j*q_r)],
        [  2*(q_i*q_j + q_k*q_r), 1 - 2*(q_i**2 + q_k**2),   2*(q_j*q_k - q_i*q_r)],
        [  2*(q_i*q_k - q_j*q_r),   2*(q_j*q_k + q_i*q_r), 1 - 2*(q_i**2 + q_j**2)]
    ])

    return trans