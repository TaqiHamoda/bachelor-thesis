import time, random, json, scipy.optimize, os
import numpy as np

from cosserat_rod.cosserat_rod import CosseratRod
from cosserat_rod.cosserat_rod_quaternion import CosseratRodQuaternion

from utils.setup_CTCR import setupCTCR
from utils.random_joints import random_joints
from utils.optimization import approx_fprime, quaternion_to_rotation, EPSILON

from solvers.newton import Newton
from solvers.broyden import Broyden

from integrators.integrators import ForwardEuler
from integrators.runge_kutta import RK2, RK3, RK4
from integrators.adams_bashforth import AB2, AB3, AB4

from data_collection.step_size import StepSizeDataCollection
from data_collection.stiffness import StiffnessDataCollection
from data_collection.root_redundancy import RootRedundancyDataCollection
from data_collection.local_configurations import LocalConfigurationsDataCollection
from data_collection.step_size_jac_step import StepSizeJacStepDataCollection

ALPHAS = []
BETAS = []
TIPS = []
RESIDUALS = []


def constructTransMatrix(pos, ori) -> np.ndarray:
    trans = np.zeros((4, 4))

    # Construct Rotation
    trans[:3, :3] = quaternion_to_rotation(ori)

    # Construct Translation
    trans[:3, 3] = pos
    trans[3, 3] = 1

    return trans


def parseDataset(file: str, dir: str):
    data = []

    with open(dir + "/" + file, "r") as f:
        l = f.readline()
        while l != "":
            l = l.split(",")
            d = {
                "alphas": [float(l[0]), float(l[2]), float(l[4])],
                "betas": [float(l[1])*1e-3, float(l[3])*1e-3, float(l[5])*1e-3],
            }

            pos_base = [float(l[12])*1e-3, float(l[13])*1e-3, float(l[14])*1e-3]
            ori_base = [float(l[15]), float(l[16]), float(l[17]), float(l[18])]

            pos_outer = [float(l[19])*1e-3, float(l[20])*1e-3, float(l[21])*1e-3]
            ori_outer = [float(l[22]), float(l[23]), float(l[24]), float(l[25])]

            pos_middle = [float(l[26])*1e-3, float(l[27])*1e-3, float(l[28])*1e-3]
            ori_middle = [float(l[29]), float(l[30]), float(l[31]), float(l[32])]

            pos_inner = [float(l[33])*1e-3, float(l[34])*1e-3, float(l[35])*1e-3]
            ori_inner = [float(l[36]), float(l[37]), float(l[38]), float(l[39])]

            t_aurora_base = constructTransMatrix(pos_base, ori_base)
            t_aurora_outer = constructTransMatrix(pos_outer, ori_outer)
            t_aurora_middle = constructTransMatrix(pos_middle, ori_middle)
            t_aurora_inner = constructTransMatrix(pos_inner, ori_inner)

            t_base_aurora = np.linalg.inv(t_aurora_base)
            t_base_outer = t_base_aurora@t_aurora_outer
            t_base_middle = t_base_aurora@t_aurora_middle
            t_base_inner = t_base_aurora@t_aurora_inner

            d["outer_position"] = t_base_outer[:3, 3].tolist()
            d["middle_position"] = t_base_middle[:3, 3].tolist()
            d["inner_position"] = t_base_inner[:3, 3].tolist()

            data.append(d)
            l = f.readline()

    with open(dir + "/Dataset.json", "w") as f:
        json.dump(data, f)

    random.shuffle(data)

    if not os.path.isdir(dir + "/dataset_randomized/"):
        os.mkdir(dir + "/dataset_randomized/")

    for i in range(1000, len(data), 1000):
        with open(dir + f"/dataset_randomized/{i//1000}.json", "w") as f:
            json.dump(data[i - 1000:i], f)


def residuals(kappas):
    loss = lambda r: np.sum(np.abs(r), axis=1)

    ctcr = setupCTCR(
        ls = [169*1e-3, 65*1e-3, 10*1e-3],
        lc = [41*1e-3, 100*1e-3, 100*1e-3],
        ro = [0.5/2*1e-3, 0.9/2*1e-3, 1.5/2*1e-3],
        ri = [0.4/2*1e-3, 0.7/2*1e-3, 1.2/2*1e-3],
        k = kappas,
        nu = 0.3,
        E = 50e09
    )

    RESIDUALS.clear()
    joints = np.zeros((3, 2))
    u_init = np.zeros((3,))

    h = 1e-3
    model = CosseratRod(ctcr, joints, Newton(), RK4(h))
    for i in range(len(ALPHAS)):
        model.ALPHAS[:] = ALPHAS[i]
        model.BETAS[:] = BETAS[i]

        model._initialize()
        model.solve(u_init)

        RESIDUALS.append(1e3*(model.backbone[2][-1][:3] - TIPS[i]))

    return loss(np.array(RESIDUALS))


def optimizeParameters(p0: list[float], dir: str, offset: list[float] = [0, 0, 0], files:int = 10, dp_per_file:int = 1000):
    for i in range(files):
        with open(dir + f"/{i + 1}.json", "r") as f:
            data = json.load(f)
            for j in range(dp_per_file):
                ALPHAS.append(data[j]["alphas"])
                BETAS.append(data[j]["betas"])
                TIPS.append(data[j]["inner_position"])

    TIPS = np.array(TIPS)
    TIPS[:, 0] -= offset[0]
    TIPS[:, 1] -= offset[1]
    TIPS[:, 2] -= offset[2]

    print("Started")

    start = time.perf_counter()
    ret = scipy.optimize.least_squares(residuals, p0, bounds=(0, np.inf))
    end = time.perf_counter()

    res = np.average(np.abs(RESIDUALS), axis=0)
    rel_err = np.linalg.norm(res)

    print(f"Runtime: {end - start} seconds")
    print(f"Optimized parameters: {ret.x}")
    print(f"Average Error X: {res[0]} mm")
    print(f"Average Error Y: {res[1]} mm")
    print(f"Average Error Z: {res[2]} mm")
    print(f"Average Error: {rel_err} mm")


if __name__ == "__main__":
    dataset_datapoint = {
    "alphas": [-1.0183003313416519, -0.26393483699307174, -0.7455252862272979],
    "betas": [
      -0.0643329313385009, -0.059065703229000704, -0.031174501501201147
    ],
    "outer_position": [
      -0.002065296705399884, -0.013878919472079476, 0.07894178956634453
    ],
    "middle_position": [
      -0.00747498746316236, -0.02888605939486194, 0.1008461638255243
    ],
    "inner_position": [
      -0.026480598324591813, -0.05527128799660408, 0.11584196819776656
    ]
  }

    ctcr = setupCTCR(
        ls = [169*1e-3, 65*1e-3, 10*1e-3],
        lc = [41*1e-3, 100*1e-3, 100*1e-3],
        ro = [0.5/2*1e-3, 0.9/2*1e-3, 1.5/2*1e-3],
        ri = [0.4/2*1e-3, 0.7/2*1e-3, 1.2/2*1e-3],
        k = [28, 12.4, 4.37],
        # k = [21.36578336, 12.47604884, 4.7387904],
        nu = 0.3,
        E = 50e09
    )

    # ctcr = setupCTCR()

    # offset = np.array([0.01816448, -0.00055646, -0.00543566])

    # joints = np.array([
    #     [ 2.52617158, -0.45257322],
    #     [ 1.44485541, -0.34109823],
    #     [-2.80050496, -0.18282406]
    # ])
    # joints[:, 0] = dataset_datapoint["alphas"]
    # joints[:, 1] = dataset_datapoint["betas"]

    
    # model_gt = CosseratRod(ctcr, joints, Newton(), RK4(1e-5))
    # print(np.linalg.norm(model_gt.evaluate(u_opt)))

    # print(1e3*(model.construct_tip()[:3, 3] - model_gt.construct_tip()[:3, 3]))

    for a in ((0, 0), (0, 1), (1, 0), (1, 1)):
        joints = np.zeros((3, 2))
        joints[0, 0] = a[1]*np.pi  # Inner tube
        joints[1, 0] = a[0]*np.pi  # Middle tube

        h = 1e-3
        model = CosseratRod(ctcr, joints, Broyden(), AB2(h))
        model.solve(np.zeros((3, )))

        with open(f"{a[0]}-{a[1]}.json", "w") as f:
            json.dump(model.backbone, f)



    # d = StiffnessDataCollection()
    # d.run()
    # d.createVisualization()