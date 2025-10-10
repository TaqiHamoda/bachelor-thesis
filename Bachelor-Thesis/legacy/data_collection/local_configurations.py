import numpy as np
from time import perf_counter
import os, json, pandas

import matplotlib.pyplot as plt

from utils.optimization import EPSILON
from utils.random_joints import random_joints
from utils.setup_CTCR import setupCTCR
from data_collection.data_collection import DataCollection

from solvers.broyden import Broyden
from integrators.adams_bashforth import AB4

from cosserat_rod.cosserat_rod import CosseratRod

class LocalConfigurationsDataCollection(DataCollection):
    def __init__(self, thread_count: int = 3) -> None:
        super().__init__("LocalConfigurations", thread_count)
        self.overwrite_data = False
        self.h = 1e-3


    def _collect_data(self, model_gt: CosseratRod, model: CosseratRod) -> dict:
        iters = np.random.randint(10, 100)

        u_opts = np.zeros((iters, model.n))
        s_iters = np.zeros((iters, ))

        start_joints = np.zeros((model.n, 2))
        start_joints[:, 0] = model.ALPHAS
        start_joints[:, 1] = model.BETAS

        end_joints = random_joints(model.n, model.ctcr)

        diff = (end_joints - start_joints)/iters

        u_init = np.zeros((model.n, ))
        joints = start_joints
        for i in range(iters):
            model = CosseratRod(model.ctcr, joints, model.solver, model.integrator)

            u_opts[i, :] = model.solve(u_init)
            s_iters[i] = len(model.solver.err_list)

            u_init = u_opts[i, :]
            joints = joints + diff

        return {
            "u_opts": u_opts.tolist(),
            "start alpha": start_joints[:, 0].tolist(),
            "end alpha": end_joints[:, 0].tolist(),
            "start beta": start_joints[:, 1].tolist(),
            "end beta": end_joints[:, 1].tolist(),
            "solver iterations": s_iters.tolist(),
            "alpha difference": diff[:, 0].tolist(),
            "beta difference": diff[:, 1].tolist()
        }


    def run(self) -> None:
        ctcr = setupCTCR()
        joints = random_joints(3, ctcr)

        n = 100
        for i in range(n):
            models = {
                "R": CosseratRod(ctcr, joints, Broyden(), AB4(self.h))
            }

            self.collect_data_parallel(models, iterations=n, filename=f"{i}", randomize=True)
            print(f"{100*(i + 1)/n}% Done")


    def createVisualization(self) -> None:
        super().createVisualization()

        data = {
            "alpha difference": [],
            "beta difference": [],
            "solver iterations": [],
        }

        for f_name in self.filenames:
            with open(f"./{self.DIR_PATH}/{self.name}/{f_name}", "r") as f:
                info: dict[str, list[dict]] = json.load(f)

                for (k, v) in info.items():
                    data["alpha difference"].extend([np.linalg.norm(vi["alpha difference"]) for vi in v])
                    data["beta difference"].extend([np.linalg.norm(vi["beta difference"]) for vi in v])
                    data["solver iterations"].extend([np.average(vi["solver iterations"][1:]) for vi in v])


        # for gk in ("alpha", "beta"):
            # plt.xlabel(f"{gk.title()} Difference")
            # plt.ylabel("Solver Iterations")
            # plt.title(f"{gk.title()} Difference vs Solver Iterations")

            # # plt.yscale("log")

            # plt.plot(data[f"{gk} difference"], data["solver iterations"], "o")
            # plt.tight_layout()

            # if not os.path.isdir(f"./{self.DIR_PATH}/plots/{self.name}"):
            #     os.mkdir(f"./{self.DIR_PATH}/plots/{self.name}")
            # plt.savefig(f"./{self.DIR_PATH}/plots/{self.name}/{gk}.png")
            # plt.close()

        data_x = data["alpha difference"]
        data_y = data["beta difference"]
        data_z = data["solver iterations"]

        x_min, x_max = 0, 0.3
        y_min, y_max = 0, 0.01

        n = 100
        x = np.linspace(x_min, x_max, n)
        y = np.linspace(y_min, y_max, n)
        z = np.zeros((n, n, 2))

        for k in range(len(data_x)):
            d_x, d_y, d_z = data_x[k], data_y[k], data_z[k]

            i = int(n*(d_x - x_min)/(x_max - x_min))
            j = int(n*(d_y - y_min)/(y_max - y_min))

            if i > n or j > n:
                continue

            if i == n or d_x < x[i]:
                i -= 1
            if j == n or d_y < y[j]:
                j -= 1

            z[j, i, 0] += d_z
            z[j, i, 1] += 1

        indx = z[:, :, 1].nonzero()
        z[:, :, 0][indx] /= z[:, :, 1][indx]
        z = z[:, :, 0]

        pcm = plt.pcolormesh(x, y, z, cmap="RdBu_r")
        c_bar = plt.colorbar(pcm)

        plt.xlabel("alpha difference")
        plt.ylabel("beta difference")
        c_bar.set_label("solver iterations")

        # plt.title(f"{d_variable.title()} (Model {mk}): {gk[g][0].title()} vs {gk[g][1].title()}")
        plt.tight_layout()

        if not os.path.isdir(f"./{self.DIR_PATH}/plots/{self.name}"):
                os.mkdir(f"./{self.DIR_PATH}/plots/{self.name}")
        plt.savefig(f"./{self.DIR_PATH}/plots/{self.name}/joints_difference.png")
        plt.close()
