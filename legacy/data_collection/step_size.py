import numpy as np
from time import perf_counter
import os, json, pandas

import matplotlib.pyplot as plt

from utils.random_joints import random_joints
from utils.setup_CTCR import setupCTCR
from data_collection.data_collection import DataCollection, SOLVERS, INTEGRATORS

from cosserat_rod.cosserat_rod import CosseratRod
from cosserat_rod.cosserat_rod_quaternion import CosseratRodQuaternion


class StepSizeDataCollection(DataCollection):
    def __init__(self, thread_count: int = 3) -> None:
        super().__init__("StepSize", thread_count)
        self.overwrite_data = True


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
            "step_size": model.integrator.h
        }


    def run(self) -> None:
        ctcr = setupCTCR()
        joints = random_joints(3, ctcr)

        start_h = 1e-4
        stop_h = 3e-3
        hs = np.linspace(start_h, stop_h, num=30)

        for h in range(hs.size):
            models = {}
            for sk in SOLVERS.keys():
                for sfk in SOLVERS[sk].keys():
                    for ik in INTEGRATORS.keys():
                        models[f"{sk}_{sfk}_{ik}_R"] = CosseratRod(ctcr, joints, SOLVERS[sk][sfk](), INTEGRATORS[ik](hs[h]))
                        models[f"{sk}_{sfk}_{ik}_Q"] = CosseratRodQuaternion(ctcr, joints, SOLVERS[sk][sfk](), INTEGRATORS[ik](hs[h]))

            self.collect_data_parallel(models, iterations=5, filename=f"{h}", randomize=True)
            print(f"{100*(h + 1)/hs.size}% Done")


    def createVisualization(self) -> None:
        super().createVisualization()

        data = {}
        for f_name in self.filenames:
            with open(f"./{self.DIR_PATH}/{self.name}/{f_name}", "r") as f:
                info: dict[str, list[dict]] = json.load(f)

                for (k, v) in info.items():
                    metadata = k.split("_")
                    
                    if data.get(metadata[0], None) is None:
                        data[metadata[0]] = {}  # Solver
                    
                    if data[metadata[0]].get(metadata[1], None) is None:
                        data[metadata[0]][metadata[1]] = {}  # Solver Flavor
                    
                    if data[metadata[0]][metadata[1]].get(metadata[2], None) is None:
                        data[metadata[0]][metadata[1]][metadata[2]] = {}  # Integrator
                    
                    if data[metadata[0]][metadata[1]][metadata[2]].get(metadata[3], None) is None:
                        data[metadata[0]][metadata[1]][metadata[2]][metadata[3]] = {  # Model Type
                            "step_size": [],
                            "residual error": [],
                            "position error (mm)": [],
                            "runtime (seconds)": []
                        }

                    data_segment = data[metadata[0]][metadata[1]][metadata[2]][metadata[3]]
                    data_segment["step_size"].append(np.average([vi["step_size"] for vi in v]))
                    data_segment["residual error"].append(np.average([vi["residual error"] for vi in v]))
                    data_segment["position error (mm)"].append(np.average([vi["position error"] for vi in v]))
                    data_segment["runtime (seconds)"].append(np.average([vi["runtime"] for vi in v]))

        gks = ("residual error", "position error (mm)", "runtime (seconds)")

        for sk in data.keys():
            for sfk in data[sk].keys():
                for ik in data[sk][sfk].keys():
                    for mk in data[sk][sfk][ik].keys():
                        for i in range(len(gks)):
                            gk = gks[i]
                            data_segment = data[sk][sfk][ik][mk]

                            plt.xlabel("Step Size (m)")
                            plt.ylabel(gk.title())
                            plt.title(f"{gk.title()} vs Step Size: {sk}-{ik} (model {mk})")

                            # plt.yscale("log")

                            plt.plot(data_segment["step_size"], data_segment[gk], "o")
                            plt.tight_layout()

                            if not os.path.isdir(f"./{self.DIR_PATH}/plots/{self.name}/{gk}"):
                                os.mkdir(f"./{self.DIR_PATH}/plots/{self.name}/{gk}")
                            plt.savefig(f"./{self.DIR_PATH}/plots/{self.name}/{gk}/{sk}_{sfk}_{ik}_{mk}.png")
                            plt.close()
