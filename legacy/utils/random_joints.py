import numpy as np

from utils.tube import Tube

def random_joints(n: int, ctcr: list[Tube], seed: float|None = None) -> np.ndarray:
    # Only seed the rng if the seed is not zero
    if seed:
        np.random.seed(seed)

    # The random joint values. The first column is the alpha values (rotation)
    # The second column is the beta values (translation)
    joints = np.zeros((n, 2))
    joints[:, 0] = (2*np.random.rand(n) - 1) * np.pi  # Sample from the interval (-pi, pi)

    # Sample Translation joints
    # l1 = ctcr[0].L
    # l2 = ctcr[1].L
    # l3 = ctcr[2].L

    # M_B =  np.array([
    #     [l2 - l1, l3 - l2, -l3],
    #     [0,       l3 - l2, -l3],
    #     [0,       0,       -l3]
    # ])

    # b1 = -np.random.rand()*ctcr[0].L  # fixed beta 1 value results in constant robot length (in meters)
    # betas = np.random.random(3)*np.abs(b1)/l1
    # betas[0] = 1

    # joints[0, 1] = b1
    # joints[1:, 1] = (M_B@betas)[1:]

    sample = -np.random.rand()*ctcr[0].L
    joints[0, 1] = sample
    
    i = 1
    while i < n:
        sample = -np.random.rand()*ctcr[i].L
        if sample < joints[i - 1, 1] or ctcr[i].L + sample > ctcr[i - 1].L + joints[i - 1, 1]:
            continue

        joints[i, 1] = sample
        i += 1

    return joints