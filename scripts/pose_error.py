import numpy as np
import scipy.optimize, scipy.stats, time, json, os
from threading import Thread
import matplotlib.pyplot as plt

from dataparser import runModel
from graphs import histogram

N = 10_000
DATA_N = 100_000

THREAD_COUNT = 8

USE_SAVED_INDICES = False

DATA_DIR = "../data/"
GRAPHS_DIR = f"{DATA_DIR}/graphs/"
GROUND_TRUTH_PATH = "CRL-Dataset-CTCR-Pose.csv"
INDICES_FILE_PATH = f"{DATA_DIR}/indices.json"

DATA = []  # Alphas, Betas, pos_base, ori_base, pos_outer, ori_outer, pos_middle, ori_middle, pos_inner, ori_inner
TIPS = np.zeros((DATA_N, 3, 3))
POS_ERR = np.zeros((DATA_N, 3))
F_CALLS = [0]

BASE_OFFSET = [0, 0, 0]
OUTER_OFFSET = [0, 0, 0]
MIDDLE_OFFSET = [0, 0, 0]
INNER_OFFSET = [0, 0, 0]

# ORIGINAL_PARAMETERS = np.array(BASE_OFFSET + OUTER_OFFSET + MIDDLE_OFFSET + INNER_OFFSET)
ORIGINAL_PARAMETERS = np.array([0.0021792646597059884, 0.01126373190601274, 0.0010055156330952485, 0.0018443188994495565, -0.00018399665953102926, 0.0005040246029058141, -0.0005226400228246249, -0.004690433186634946, -0.0013207434413336808, -0.0035009435363276115, -0.006389302059844599, -0.0001766858857704401])

def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    # Reference: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    q_r, q_i, q_j, q_k = q

    rot = np.array([
        [1 - 2*(q_j**2 + q_k**2),   2*(q_i*q_j - q_k*q_r),   2*(q_i*q_k + q_j*q_r)],
        [  2*(q_i*q_j + q_k*q_r), 1 - 2*(q_i**2 + q_k**2),   2*(q_j*q_k - q_i*q_r)],
        [  2*(q_i*q_k - q_j*q_r),   2*(q_j*q_k + q_i*q_r), 1 - 2*(q_i**2 + q_j**2)]
    ])

    return rot/np.linalg.norm(q)


def parseDataset(filepath: str):
    alphas = []
    betas = []

    pos_base = []
    ori_base = []

    pos_outer = []
    ori_outer = []

    pos_middle = []
    ori_middle = []

    pos_inner = []
    ori_inner = []

    n = 0
    with open(filepath, "r") as f:
        l = f.readline()
        while l != "":
            l = l.split(",")

            alphas.append((float(l[0]), float(l[2]), float(l[4])))
            betas.append((float(l[1])*1e-3, float(l[3])*1e-3, float(l[5])*1e-3))

            pos_base.append((float(l[12])*1e-3, float(l[13])*1e-3, float(l[14])*1e-3))
            ori_base.append((float(l[15]), float(l[16]), float(l[17]), float(l[18])))

            pos_outer.append((float(l[19])*1e-3, float(l[20])*1e-3, float(l[21])*1e-3))
            ori_outer.append((float(l[22]), float(l[23]), float(l[24]), float(l[25])))

            pos_middle.append((float(l[26])*1e-3, float(l[27])*1e-3, float(l[28])*1e-3))
            ori_middle.append((float(l[29]), float(l[30]), float(l[31]), float(l[32])))

            pos_inner.append((float(l[33])*1e-3, float(l[34])*1e-3, float(l[35])*1e-3))
            ori_inner.append((float(l[36]), float(l[37]), float(l[38]), float(l[39])))

            n += 1

            l = f.readline()

    return [np.array(alphas), np.array(betas), np.array(pos_base), np.array(ori_base), np.array(pos_outer), np.array(ori_outer), np.array(pos_middle), np.array(ori_middle), np.array(pos_inner), np.array(ori_inner)]


def saveData(p0: np.ndarray):
    F_CALLS[0] += 1

    p_err = 1e3*np.linalg.norm(POS_ERR[:N], axis=1)
    with open(f"{DATA_DIR}/least_squares.json", "w") as f:
        json.dump({
            "Function Calls": F_CALLS[0],
            "Optimized parameters": p0.tolist(),
            "Average Position Error (mm)": np.average(p_err),
            "Median Position Error (mm)": np.median(p_err),
            "STD Position Error (mm)": np.std(p_err),
            "Min Position Error (mm)": np.min(p_err),
            "Max Position Error (mm)": np.max(p_err),
            "Position Errors (mm)": p_err.tolist(),
        }, f)

    histogram(p_err)

    plt.tight_layout()
    plt.savefig(f"{GRAPHS_DIR}/dataset_histogram.png")
    plt.close()


def constructTips():
    def _constructTips(i, j):
        for k in range(i, j):
            dp = runModel("Broyden", "AB2", 1e-3, quaternion=True, without_check=True, alphas=DATA[0][k], betas=DATA[1][k], forces=[0, 0, 0])

            TIPS[k][0] = dp.tip_positions[0]
            TIPS[k][1] = dp.tip_positions[1]
            TIPS[k][2] = dp.tip_positions[2]


    ts: list[Thread] = []

    step = N//THREAD_COUNT
    for h in range(0, N, step):
        if h + step < N:
            ts.append(Thread(target=_constructTips, args=(h, h + step)))
        else:
            ts.append(Thread(target=_constructTips, args=(h, N)))

        ts[-1].start()

    for t in ts:
        t.join()


def residuals(parameters: np.ndarray):
    res = np.zeros((N,))

    constructTips()

    delta_pos_base = parameters[0:3]
    delta_pos_outer = parameters[3:6]
    delta_pos_middle = parameters[6:9]
    delta_pos_inner = parameters[9:12]

    t_aurora_base = np.zeros((4, 4))
    t_aurora_outer = np.zeros((4, 4))
    t_aurora_middle = np.zeros((4, 4))
    t_aurora_inner = np.zeros((4, 4))

    t_aurora_base[3, 3] = 1
    t_aurora_outer[3, 3] = 1
    t_aurora_middle[3, 3] = 1
    t_aurora_inner[3, 3] = 1
    for i in range(N):
        pos_base = DATA[2][i]
        ori_base = DATA[3][i]

        pos_outer = DATA[4][i]
        ori_outer = DATA[5][i]

        pos_middle = DATA[6][i]
        ori_middle = DATA[7][i]

        pos_inner = DATA[8][i]
        ori_inner = DATA[9][i]

        t_aurora_base[:3, :3] = quaternion_to_rotation(ori_base)
        t_aurora_base[:3, 3] = pos_base + delta_pos_base

        t_aurora_outer[:3, :3] = quaternion_to_rotation(ori_outer)
        t_aurora_outer[:3, 3] = pos_outer + delta_pos_outer

        t_aurora_middle[:3, :3] = quaternion_to_rotation(ori_middle)
        t_aurora_middle[:3, 3] = pos_middle + delta_pos_middle

        t_aurora_inner[:3, :3] = quaternion_to_rotation(ori_inner)
        t_aurora_inner[:3, 3] = pos_inner + delta_pos_inner

        t_base_aurora = np.linalg.inv(t_aurora_base)
        t_base_outer = t_base_aurora@t_aurora_outer
        t_base_middle = t_base_aurora@t_aurora_middle
        t_base_inner = t_base_aurora@t_aurora_inner

        res[i] = np.linalg.norm(1e3*(t_base_outer[:3, 3] - TIPS[i][2]))
        res[i] += np.linalg.norm(1e3*(t_base_middle[:3, 3] - TIPS[i][1]))
        res[i] += np.linalg.norm(1e3*(t_base_inner[:3, 3] - TIPS[i][0]))

        POS_ERR[i] = t_base_inner[:3, 3] - TIPS[i][0]

    saveData(parameters)

    return np.sum(res)


def optimizeParameters(p0: list[float]):
    print("Started")

    start = time.perf_counter()
    ret = scipy.optimize.least_squares(residuals, p0, method="trf", jac="3-point", verbose=2)
    end = time.perf_counter()

    print(f"Runtime: {end - start} seconds")


def indicesWithoutOutliers():
    indices = np.zeros((DATA_N, )) == 0
    for a in range(3):  # Determine outliers along each axis
        pos_err = POS_ERR[:, a]

        Q1 = np.percentile(pos_err, 25, method='midpoint')
        Q3 = np.percentile(pos_err, 75, method='midpoint')
        IQR = Q3 - Q1

        # Create outlier bounds
        lower_limit = Q1 - 1.5*IQR
        upper_limit = Q3 + 1.5*IQR

        indices = np.logical_and(indices, np.logical_and(pos_err > lower_limit, pos_err < upper_limit))

    return np.arange(DATA_N)[indices]


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    DATA = parseDataset(GROUND_TRUTH_PATH)
    print("Parsed Data\n")

    if not USE_SAVED_INDICES:
        with open(INDICES_FILE_PATH, "w") as f:
            residuals(ORIGINAL_PARAMETERS)
            json.dump(POS_ERR.tolist(), f)

    indices = np.arange(DATA_N)
    with open(INDICES_FILE_PATH, "r") as f:
        POS_ERR = np.array(json.load(f))
        indices = indicesWithoutOutliers()
        print(indices.size)

    np.random.shuffle(indices)
    indices = indices[:N]

    for i in range(len(DATA)):
        DATA[i] = DATA[i][indices]

    optimizeParameters(ORIGINAL_PARAMETERS)

    with open(f"{DATA_DIR}/least_squares.json", 'r') as f:
        d = json.load(f)
        p_err = d["Position Errors (mm)"]
        ecdf = scipy.stats.ecdf(p_err).cdf
        print(f"Probability that values are less than 6.3 mm: {ecdf.evaluate(6.3) * 100}%")  # Should be around 92.67%