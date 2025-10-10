import os, json, time
from multiprocessing import Process
import numpy as np

from utils.random_joints import random_joints
from cosserat_rod.cosserat_rod import CosseratRod

from solvers.newton import Newton, NewtonMomentum, NewtonNesterov, NewtonAdam, NewtonAdamW
from solvers.broyden import Broyden, BroydenMomentum, BroydenNesterov, BroydenAdam, BroydenAdamW

from integrators.integrators import ForwardEuler
from integrators.runge_kutta import RK2, RK3, RK4
from integrators.adams_bashforth import AB2, AB3, AB4


SOLVERS = {
    "Newton": {
        "Vanilla": Newton,
        "Momentum": NewtonMomentum,
        "Nesterov": NewtonNesterov,
        "Adam": NewtonAdam,
        "AdamW": NewtonAdamW
    },
    "Broyden": {
        "Vanilla": Broyden,
        "Momentum": BroydenMomentum,
        "Nesterov": BroydenNesterov,
        "Adam": BroydenAdam,
        "AdamW": BroydenAdamW
    },
}

INTEGRATORS = {
    "Forward Euler": ForwardEuler,
    "AB2": AB2,
    "AB3": AB3,
    "AB4": AB4,
    "RK2": RK2,
    "RK3": RK3,
    "RK4": RK4
}

class DataCollection:
    DIR_PATH = "./data"
    STEP_SIZE = 5e-5

    def __init__(self, name: str, thread_count: int = 3) -> None:
        self.name = name

        self.filenames: list[str] = []

        self.thread_count = thread_count
        self.thread_store: list[Process] = []

        self.overwrite_data = False


    def _collect_data(self, model_gt: CosseratRod, model: CosseratRod) -> dict:
        raise NotImplementedError


    def collect_data(self, models: dict[str, CosseratRod], iterations: int = 10, filename: str = "data", randomize: bool = True) -> None:
        if not self.overwrite_data and os.path.isfile(f"{self.DIR_PATH}/{self.name}/{filename}.json"):
            return
        
        np.random.seed(time.perf_counter_ns()%(2**32))  # Seed the RNG in case there are multiple processes ongoing

        data = {}

        print("Started Data Collection")
        for i in range(iterations):
            joints = None  # Make sure all different models are evaluated using the same configuration
            for mk in models.keys():
                if data.get(mk, None) is None:
                    data[mk] = []

                model = models[mk]

                if joints is None:
                    if randomize:
                        joints = random_joints(model.n, model.ctcr)  # Sample random joints
                    else:
                        joints = np.zeros((model.n, 2))

                        joints[:, 0] = model.ALPHAS
                        joints[:, 1] = model.BETAS

                model.ALPHAS = joints[:, 0]
                model.BETAS = joints[:, 1]

                model_gt = CosseratRod(model.ctcr, joints, Newton(), RK4(self.STEP_SIZE))
                model.tube_lengths = model_gt.tube_lengths

                try:
                    data[mk].append(self._collect_data(model_gt, model))
                except Exception as e:
                    print(e)

                print(f"Done with model {mk}")
            print(f"Done with all models, iteration: {i + 1}")

        if not os.path.isdir(self.DIR_PATH):
            os.mkdir(self.DIR_PATH)

        if not os.path.isdir(f"{self.DIR_PATH}/{self.name}"):
            os.mkdir(f"{self.DIR_PATH}/{self.name}")

        with open(f"{self.DIR_PATH}/{self.name}/{filename}.json", "w") as f:
            json.dump(data, f)


    def collect_data_parallel(self, models: dict[str, CosseratRod], iterations: int = 10, filename: str = "data", randomize: bool = True):
        if len(self.thread_store) == self.thread_count:
            p = None
            for i in range(self.thread_count):
                if not self.thread_store[i].is_alive():
                    p = self.thread_store.pop(i)
                    break

            if p is None:
                p = self.thread_store.pop(0)

            p.join()

        p = Process(target=self.collect_data, args=(models, iterations, filename, randomize))
        p.start()
        self.thread_store.append(p)


    def run(self) -> None:
        raise NotImplementedError


    def createVisualization(self) -> None:
        if not os.path.isdir(self.DIR_PATH):
            os.mkdir(self.DIR_PATH)

        if not os.path.isdir(f"{self.DIR_PATH}/plots"):
            os.mkdir(f"{self.DIR_PATH}/plots")

        if not os.path.isdir(f"{self.DIR_PATH}/plots/{self.name}"):
            os.mkdir(f"{self.DIR_PATH}/plots/{self.name}")

        print("Begin creating plots")
        self.filenames = os.listdir(f"./{self.DIR_PATH}/{self.name}")
