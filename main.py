import numpy as np

import yaml, subprocess, os, urllib.request
from multiprocessing import Pool

from .utils.dataparser import runModel
from .utils.visualizations import process_datafiles

GROUND_TRUTH_POSE = "CRL-Dataset-CTCR-Pose.csv"
GROUND_TRUTH_POSE_URL = "https://raw.githubusercontent.com/ContinuumRoboticsLab/CRL-Dataset-CTCR-Pose/refs/heads/main/CRL-Dataset-CTCR-Pose.csv"


def stiffness(
    data_dir,
    prefix,
    samples,
    solver,
    integrator,
    h,
    bin_path
):
    datapoints = []
    for _ in range(samples):
        datapoints.append(runModel(
            solver=solver,
            integrator=integrator,
            h=h,
            bin_path=bin_path
        ))

    with open(f"{data_dir}/{prefix}.txt", 'w') as f:
        for dp in datapoints:
            f.write(str(dp))


if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    bin_path = config['bin_path']

    data_dir = config['data_dir']
    os.makedirs(data_dir, exist_ok=True)

    stiffness_dir = f"{data_dir}/stiffness"
    os.makedirs(stiffness_dir, exist_ok=True)

    if not os.path.exists(bin_path) or not os.path.isfile(bin_path):
        subprocess.check_output(("make", ),)
        subprocess.check_output(("make", "clean"),)

    if config['runtime']['enabled']:
        with open(f"{data_dir}/{config['runtime']['data_file']}", 'w') as f:
            for solver in config['runtime']['solvers']:
                for integrator in config['runtime']['integrators']:
                    # Rotation Matrix
                    dp = runModel(
                        solver=solver,
                        integrator=integrator,
                        h=config['runtime']['h'],
                        samples=config['runtime']['samples'],
                        bin_path=bin_path
                    )
                    f.write(str(dp))

                    # Quaternion
                    dp = runModel(
                        solver=solver,
                        integrator=integrator,
                        h=config['runtime']['h'],
                        samples=config['runtime']['samples'],
                        quaternion=True,
                        bin_path=bin_path
                    )
                    f.write(str(dp))

    if config['stiffness']['enabled']:
        with Pool(processes=config['stiffness']['threads']) as pool:
            pool.map(
                stiffness,
                [(stiffness_dir, f"stiffness{i}", config['stiffness']['samples_per_thread'], config['stiffness']['solver'], config['stiffness']['integrator'], config['stiffness']['h'], bin_path) for i in range(config['stiffness']['threads'])]
            )

    if config['finite_differences']['enabled']:
        with open(f"{data_dir}/finite_differences.txt", 'w') as f:
            j_start = config['finite_differences']['j_start']
            j_step = config['finite_differences']['j_step']
            j_end = config['finite_differences']['j_end']

            for j in np.arange(j_start, j_end + j_step/2, j_step):
                j_val = float(f"{j}e-3")
                
                # Rotation Matrix
                dp = runModel(
                    solver=config['finite_differences']['solver'],
                    integrator=config['finite_differences']['integrator'],
                    h=config['finite_differences']['h'],
                    samples=config['finite_differences']['samples'],
                    fd_step=j_val,
                    bin_path=bin_path
                )
                f.write(str(dp))

                # Quaternion
                dp = runModel(
                    solver=config['finite_differences']['solver'],
                    integrator=config['finite_differences']['integrator'],
                    h=config['finite_differences']['h'],
                    samples=config['finite_differences']['samples'],
                    fd_step=j_val,
                    quaternion=True,
                    bin_path=bin_path
                )
                f.write(str(dp))

    if config['visualizations']['enabled']:
        process_datafiles(
            stiffness_dir=stiffness_dir,
            data_dir=data_dir
        )

    if config['error_evaluation']['enabled']:
        if not os.path.exists(GROUND_TRUTH_POSE) or not os.path.isfile(GROUND_TRUTH_POSE):
            urllib.request.urlretrieve(GROUND_TRUTH_POSE_URL, GROUND_TRUTH_POSE)

        


