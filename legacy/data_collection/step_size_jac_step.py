import numpy as np
from time import perf_counter
import os, json, pandas

import matplotlib.pyplot as plt

from utils.random_joints import random_joints
from utils.setup_CTCR import setupCTCR
from data_collection.data_collection import DataCollection

from solvers.broyden import Broyden
from integrators.adams_bashforth import AB4

from cosserat_rod.cosserat_rod import CosseratRod


class StepSizeJacStepDataCollection(DataCollection):
    def __init__(self, thread_count: int = 3) -> None:
        super().__init__("StepSizeJacStep", thread_count)
        self.overwrite_data = False


    def _collect_data(self, model_gt: CosseratRod, model: CosseratRod) -> dict:
        u_init_guess = np.zeros((3,))
        u_init_opt = np.array((np.NAN, np.NAN, np.NAN))

        start_time = perf_counter()
        try:
            u_init_opt = model.solve(u_init_guess[:])
        except Exception as e:
            print(e)
        end_time = perf_counter()

        residual_error = np.linalg.norm(model_gt.evaluate(u_init_opt[:]))

        tip = model.construct_tip()
        tip_ground_truth = model_gt.construct_tip()

        position_error = 1e3*np.linalg.norm(tip[:3, 3] - tip_ground_truth[:3, 3])
        orientation_error = (np.trace(tip_ground_truth[:3, :3].T@tip[:3, :3]) - 1)/2

        return {
            "runtime": end_time - start_time,
            "function call #": model._shooting_iter,
            "error per iteration": model.solver.err_list,
            "u_guess": u_init_guess.tolist(),
            "u_opt": u_init_opt.tolist(),
            "alphas": model.ALPHAS.tolist(),
            "betas": model.BETAS.tolist(),
            "residual error": residual_error,
            "position error": position_error,
            "orientation error": orientation_error,
            "step_size": model.solver.jac_step
        }


    def run(self) -> None:
        ctcr = setupCTCR()
        joints = random_joints(3, ctcr)

        start_h = 1e-3
        stop_h = 1e-2
        hs = np.linspace(start_h, stop_h, num=30)

        for h in range(hs.size):
            models = {
                "R": CosseratRod(ctcr, joints, Broyden(jac_step=hs[h]), AB4(1e-3))
            }

            self.collect_data_parallel(models, iterations=100, filename=f"{h}", randomize=True)
            print(f"{100*(h + 1)/hs.size}% Done")


    def createVisualization(self) -> None:
        super().createVisualization()

        data = {
            "step_size": [],
            "residual error": [],
            "position error (mm)": [],
            "runtime (seconds)": [],
            "solver iterations": []
        }
        for f_name in self.filenames:
            with open(f"./{self.DIR_PATH}/{self.name}/{f_name}", "r") as f:
                info: dict[str, list[dict]] = json.load(f)

                for (k, v) in info.items():
                    data["step_size"].append(np.average([vi["step_size"] for vi in v]))
                    data["residual error"].append(np.average([vi["residual error"] for vi in v]))
                    data["position error (mm)"].append(np.average([vi["position error"] for vi in v]))
                    data["runtime (seconds)"].append(np.average([vi["runtime"] for vi in v]))
                    data["solver iterations"].append(np.average([len(vi["error per iteration"]) for vi in v]))

        gks = ("residual error", "position error (mm)", "runtime (seconds)", "solver iterations")

        for i in range(len(gks)):
            gk = gks[i]

            plt.xlabel("Jacobian Step Size (m)")
            plt.ylabel(gk.title())
            plt.title(f"{gk.title()} vs Jacobian Step Size")

            # plt.yscale("log")

            plt.plot(data["step_size"], data[gk], "o")
            plt.tight_layout()

            if not os.path.isdir(f"./{self.DIR_PATH}/plots/{self.name}"):
                os.mkdir(f"./{self.DIR_PATH}/plots/{self.name}")
            plt.savefig(f"./{self.DIR_PATH}/plots/{self.name}/{gk}.png")
            plt.close()
