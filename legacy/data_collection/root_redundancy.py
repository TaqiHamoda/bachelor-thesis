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


class RootRedundancyDataCollection(DataCollection):
    def __init__(self, thread_count: int = 3) -> None:
        super().__init__("RootRedundancy", thread_count)
        self.overwrite_data = False
        self.iters = 100
        self.h = 1e-3


    def _collect_data(self, model_gt: CosseratRod, model: CosseratRod) -> dict:
        f = lambda x: 0 if x.size == 0 else 1 + f(x[np.linalg.norm(x - x[0, :], axis=1) > EPSILON])  # Algorithm to detect # of unique solutions

        u_opts = np.zeros((self.iters, 3))

        for i in range(self.iters):
            u_init_guess = 20*np.random.random((model.n, )) - 10  # Sample points between [-10, 10]
            u_opts[i, :] = model.solve(u_init_guess[:])

        return {
            "u_opts": u_opts.tolist(),
            "alphas": model.ALPHAS.tolist(),
            "betas": model.BETAS.tolist(),
            "unique roots": f(u_opts)
        }


    def run(self) -> None:
        ctcr = setupCTCR()
        joints = random_joints(3, ctcr)

        for i in range(self.iters):
            models = {
                "R": CosseratRod(ctcr, joints, Broyden(), AB4(self.h))
            }

            self.collect_data_parallel(models, iterations=self.iters, filename=f"{i}", randomize=True)
            print(f"{100*(i + 1)/self.iters}% Done")


    def createVisualization(self) -> None:
        super().createVisualization()

        data = {
            "beta 1": [],
            "beta 2": [],
            "beta 3": [],
            "alpha 1": [],
            "alpha 2": [],
            "alpha 3": [],
            "unique roots": []
        }

        for f_name in self.filenames:
            with open(f"./{self.DIR_PATH}/{self.name}/{f_name}", "r") as f:
                info: dict[str, list[dict]] = json.load(f)

                for (k, v) in list(info.items()):
                    for vi in v:
                        data["beta 1"].append(vi["betas"][0])
                        data["beta 2"].append(vi["betas"][1])
                        data["beta 3"].append(vi["betas"][2])
                        data["alpha 1"].append(vi["alphas"][0])
                        data["alpha 2"].append(vi["alphas"][1])
                        data["alpha 3"].append(vi["alphas"][2])
                        data["unique roots"].append(vi["unique roots"])


        for joint in ("alpha", "beta"):
            gk = ((f"{joint} 1", f"{joint} 2"), (f"{joint} 1", f"{joint} 3"), (f"{joint} 2", f"{joint} 3"))
            for g in range(len(gk)):
                data_x = data[gk[g][0]]
                data_y = data[gk[g][1]]
                data_z = data["unique roots"]

                x_min, x_max = np.min(data_x), np.max(data_x)
                y_min, y_max = np.min(data_y), np.max(data_y)

                n = 100
                x = np.linspace(x_min, x_max, n)
                y = np.linspace(y_min, y_max, n)
                z = np.zeros((n, n, 2))

                for k in range(len(data_x)):
                    d_x, d_y, d_z = data_x[k], data_y[k], data_z[k]

                    i = int(n*(d_x - x_min)/(x_max - x_min))
                    j = int(n*(d_y - y_min)/(y_max - y_min))

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

                plt.xlabel(gk[g][0].title())
                plt.ylabel(gk[g][1].title())
                c_bar.set_label("Number of Unique Roots")

                plt.title(f"Number of Unique Roots: {gk[g][0].title()} vs {gk[g][1].title()}")
                plt.tight_layout()

                if not os.path.isdir(f"./{self.DIR_PATH}/plots/{self.name}/joints"):
                        os.mkdir(f"./{self.DIR_PATH}/plots/{self.name}/joints")
                plt.savefig(f"./{self.DIR_PATH}/plots/{self.name}/joints/{gk[g][0]}-{gk[g][1]}.png")
                plt.close()