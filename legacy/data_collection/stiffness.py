import numpy as np
import matplotlib.pyplot as plt
import os, json

from utils.optimization import approx_fprime, EPSILON
from utils.random_joints import random_joints
from utils.setup_CTCR import setupCTCR

from solvers.broyden import Broyden
from integrators.adams_bashforth import AB4

from data_collection.data_collection import DataCollection
from cosserat_rod.cosserat_rod import CosseratRod
from cosserat_rod.cosserat_rod_quaternion import CosseratRodQuaternion


class StiffnessDataCollection(DataCollection):
    def __init__(self, thread_count: int = 3) -> None:
        super().__init__("Stiffness", thread_count)
        self.h = 1e-3
        self.cutoff = 0
        self.overwrite_data = False


    def _collect_data(self, model_gt: CosseratRod, model: CosseratRod) -> dict:
        t = -np.inf

        class JacobianModel(CosseratRod):
            def evaluate(self, x: np.ndarray) -> np.ndarray:
                return model.derive(t, x)

        model.idx = 0

        joints = np.zeros((model.n, 2))
        joints[:, 0] = model.ALPHAS
        joints[:, 1] = model.BETAS

        j_model = JacobianModel(model.ctcr, joints, model.solver, model.integrator)

        u_init = 20*np.random.random((j_model.n, )) - 10  # Sample points between [-10, 10]

        psi_init = model.ALPHAS - u_init*model.BETAS
        p_rot_init = model._construct_p_rot(psi_init[0])

        y = np.concatenate((p_rot_init, u_init, psi_init, (0,)))
        f_x = model.derive(t, y)
        jac = approx_fprime(j_model, y, f_x)

        eigenvals = np.real(np.linalg.eigvals(jac))

        is_stable = bool(np.max(eigenvals) <= self.cutoff)
        condition = np.linalg.norm(y)*np.linalg.norm(jac)/np.linalg.norm(f_x)

        pos_eigenvals = np.abs(eigenvals[eigenvals.nonzero()[0]])
        if len(pos_eigenvals) == 0:
            stiff_ratio = np.NAN
        else:
            stiff_ratio = np.max(pos_eigenvals)/np.min(pos_eigenvals)

        model.solve(np.zeros((3,)))

        return {
            "u": u_init.tolist(),
            "alphas": model.ALPHAS.tolist(),
            "betas": model.BETAS.tolist(),
            "eigenvals": eigenvals.tolist(),
            "condition": condition,
            "stiff_ratio": stiff_ratio,
            "is_stable": is_stable,
            "solver iterations": len(model.solver.err_list)
        }


    def run(self) -> None:
        ctcr = setupCTCR()

        n = 100
        for i in range(n):
            joints = random_joints(3, ctcr)
            models = {
                "R": CosseratRod(ctcr, joints, Broyden(), AB4(self.h)),
                "Q": CosseratRodQuaternion(ctcr, joints, Broyden(), AB4(self.h))
            }

            self.collect_data_parallel(models, iterations=10000, filename=f"{i}", randomize=True)
            print(f"{100*(i + 1)/n}% Done")


    def createVisualization(self) -> None:
        super().createVisualization()

        data = {}
        for f_name in self.filenames:
            with open(f"./{self.DIR_PATH}/{self.name}/{f_name}", "r") as f:
                info: dict[str, list[dict]] = json.load(f)

                for (k, v) in info.items():
                    if data.get(f"{k}_stable", None) is None:
                        data[f"{k}_stable"] = {  # Model Type
                            "max eigen": [],
                            "min eigen": [],
                            "avg eigen": [],
                            "stiffness ratio": [],
                            "condition number": [],
                            "norm of initial forces": [],
                            "is stable": [],
                            "beta 1": [],
                            "beta 2": [],
                            "beta 3": [],
                            "alpha 1": [],
                            "alpha 2": [],
                            "alpha 3": [],
                            "solver iterations": []
                        }

                        data[f"{k}_unstable"] = {  # Model Type
                            "max eigen": [],
                            "min eigen": [],
                            "avg eigen": [],
                            "stiffness ratio": [],
                            "condition number": [],
                            "norm of initial forces": [],
                            "is stable": [],
                            "beta 1": [],
                            "beta 2": [],
                            "beta 3": [],
                            "alpha 1": [],
                            "alpha 2": [],
                            "alpha 3": [],
                            "solver iterations": []
                        }

                    for vi in v:
                        data_segment = None
                        if vi["is_stable"]:
                            data_segment = data[f"{k}_stable"]
                        else:
                            data_segment = data[f"{k}_unstable"]

                        data_segment["max eigen"].append(np.max(vi["eigenvals"]))
                        data_segment["min eigen"].append(np.min(vi["eigenvals"]))
                        data_segment["avg eigen"].append(np.average(vi["eigenvals"]))
                        data_segment["stiffness ratio"].append(vi["stiff_ratio"])
                        data_segment["condition number"].append(vi["condition"])
                        data_segment["norm of initial forces"].append(np.linalg.norm(vi["u"]))
                        data_segment["beta 1"].append(vi["betas"][0])
                        data_segment["beta 2"].append(vi["betas"][1])
                        data_segment["beta 3"].append(vi["betas"][2])
                        data_segment["alpha 1"].append(vi["alphas"][0])
                        data_segment["alpha 2"].append(vi["alphas"][1])
                        data_segment["alpha 3"].append(vi["alphas"][2])
                        data_segment["solver iterations"].append(vi["solver iterations"])

        dependent_vars = {
            "max eigen": (None, None, 'symlog', None, None, 'linear'),
            "min eigen": (None, None, 'symlog', None, None, 'linear'),
            "stiffness ratio": (1e-1, 1e20, 'log', 1e-1, 1e20, 'log'),
            "condition number": (0, 35, 'linear', 0, 35, 'linear')
        }

        for mk in ("R", "Q"):
            for (dk, dv) in dependent_vars.items():
                fig, axs = plt.subplots(2)

                # Stable graph
                axs[0].plot(data[f"{mk}_stable"]["norm of initial forces"], data[f"{mk}_stable"][dk], 'o')#, alpha=0.0125, markeredgecolor="none")

                axs[0].set_ylabel(f"Stable IVP")
                axs[0].set_yscale(dv[2])

                left, right = axs[0].get_ylim()
                if dv[0] is not None:
                    left = dv[0]
                if dv[1] is not None:
                    right = dv[1]

                axs[0].set_ylim((left, right))

                # Unstable graph
                axs[1].plot(data[f"{mk}_unstable"]["norm of initial forces"], data[f"{mk}_unstable"][dk], 'o')#, alpha=0.0125, markeredgecolor="none")

                axs[1].set_ylabel(f"Unstable IVP")
                axs[1].set_yscale(dv[5])

                left, right = axs[1].get_ylim()
                if dv[3] is not None:
                    left = dv[3]
                if dv[4] is not None:
                    right = dv[4]

                axs[1].set_ylim((left, right))

                fig.suptitle(f"Norm of Initial Forces Vs {dk.title()} (Model {mk})")
                fig.supxlabel("Norm of Initial Forces")
                fig.tight_layout()

                if not os.path.isdir(f"./{self.DIR_PATH}/plots/{self.name}/forces"):
                    os.mkdir(f"./{self.DIR_PATH}/plots/{self.name}/forces")
                plt.savefig(f"./{self.DIR_PATH}/plots/{self.name}/forces/{mk}_{dk}.png")
                plt.close()


            # Histogram of Eigenvalues
            avg_eigen = data[f"{mk}_stable"]["avg eigen"] + data[f"{mk}_unstable"]["avg eigen"]
            n, bins, patches = plt.hist(avg_eigen, 100)

            # add a 'best fit' line
            # mu = np.mean(avg_eigen)
            # sigma = np.std(avg_eigen)
            # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
            # plt.plot(bins, y, "--")

            plt.xlabel("Eigenvalues of IVP")
            plt.ylabel("# of Occurrences")

            plt.title(f"Histogram of Eigenvalues of the IVP (model {mk})")
            plt.tight_layout()

            if not os.path.isdir(f"./{self.DIR_PATH}/plots/{self.name}/eigen"):
                os.mkdir(f"./{self.DIR_PATH}/plots/{self.name}/eigen")
            plt.savefig(f"./{self.DIR_PATH}/plots/{self.name}/eigen/{mk}.png")
            plt.close()

            # Color Mesh Graphs
            for d_variable in ("max eigen", "solver iterations"):
                for joint in ("alpha", "beta"):
                    gk = ((f"{joint} 1", f"{joint} 2"), (f"{joint} 1", f"{joint} 3"), (f"{joint} 2", f"{joint} 3"))
                    for g in range(len(gk)):
                        data_x = data[f"{mk}_stable"][gk[g][0]] + data[f"{mk}_unstable"][gk[g][0]]
                        data_y = data[f"{mk}_stable"][gk[g][1]] + data[f"{mk}_unstable"][gk[g][1]]
                        data_z = data[f"{mk}_stable"][d_variable] + data[f"{mk}_unstable"][d_variable]

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

                        # Get the averages
                        indx = z[:, :, 1].nonzero()
                        z[:, :, 0][indx] /= z[:, :, 1][indx]

                        # Overwrite untouched areas with None
                        indx = np.where(z[:, :, 1] == 0)
                        z[:, :, 0][indx] = None

                        z = z[:, :, 0]

                        pcm = plt.pcolormesh(x, y, z, cmap="RdBu_r")
                        c_bar = plt.colorbar(pcm)

                        plt.xlabel(gk[g][0].title())
                        plt.ylabel(gk[g][1].title())
                        c_bar.set_label(d_variable.title())

                        plt.title(f"{d_variable.title()} (Model {mk}): {gk[g][0].title()} vs {gk[g][1].title()}")
                        plt.tight_layout()

                        if not os.path.isdir(f"./{self.DIR_PATH}/plots/{self.name}/joints"):
                                os.mkdir(f"./{self.DIR_PATH}/plots/{self.name}/joints")
                        plt.savefig(f"./{self.DIR_PATH}/plots/{self.name}/joints/{mk}_{d_variable}_{gk[g][0]}-{gk[g][1]}.png")
                        plt.close()