import numpy as np
import yaml
import subprocess
import os
import tqdm
import urllib.request
from multiprocessing import Process

from utils.dataparser import runModel
from utils.visualizations import process_datafiles
from utils.pose_calibration import calibrate_pose

# Constants for ground truth pose data
GROUND_TRUTH_POSE = "CRL-Dataset-CTCR-Pose.csv"
GROUND_TRUTH_POSE_URL = "https://raw.githubusercontent.com/ContinuumRoboticsLab/CRL-Dataset-CTCR-Pose/refs/heads/main/CRL-Dataset-CTCR-Pose.csv"


def stiffness(
    data_dir: str,
    prefix: str,
    samples: int,
    solver: str,
    integrator: str,
    h: float,
    bin_path: str
) -> None:
    """Runs stiffness simulations and saves results to a file.

    Args:
        data_dir: Directory to save the output file.
        prefix: Prefix for the output filename.
        samples: Number of simulation samples to run.
        solver: Solver type for the simulation.
        integrator: Integrator type for the simulation.
        h: Step size for numerical integration.
        bin_path: Path to the executable binary.
    """
    # Run simulations and collect data points
    datapoints = []
    for i in range(samples):
        dp = runModel(
            solver=solver,
            integrator=integrator,
            h=h,
            bin_path=bin_path
        )
        datapoints.append(dp)

    # Save results to file
    output_file = f"{data_dir}/{prefix}.txt"
    with open(output_file, 'w') as f:
        for dp in datapoints:
            f.write(str(dp))


if __name__ == "__main__":
    print("Loading configuration from config.yaml")
    # Load configuration from YAML file
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    bin_path = config['bin_path']
    print(f"Using binary path: {bin_path}")

    # Create directories for data and graphs
    data_dir = config['data_dir']
    os.makedirs(data_dir, exist_ok=True)

    graphs_dir = f"{data_dir}/graphs"
    os.makedirs(graphs_dir, exist_ok=True)

    stiffness_dir = f"{data_dir}/stiffness"
    os.makedirs(stiffness_dir, exist_ok=True)

    # Compile binary if it doesn't exist
    if not os.path.exists(bin_path) or not os.path.isfile(bin_path):
        print("Compiling binary...")
        subprocess.check_output(("make",))
        subprocess.check_output(("make", "clean"))
        print("Binary compilation completed")

    # Runtime experiments
    if config['runtime']['enabled']:
        print("Starting runtime experiments")
        output_file = f"{data_dir}/{config['runtime']['data_file']}"
        print(f"Preparing to write runtime results to {output_file}")
        with open(output_file, 'w') as f:
            # Generate all solver-integrator combinations
            combos = [(s, i) for s in config['runtime']['solvers'] for i in config['runtime']['integrators']]
            for solver, integrator in tqdm.tqdm(combos, desc="Running runtime experiments"):
                # Run simulation with rotation matrix
                dp = runModel(
                    solver=solver,
                    integrator=integrator,
                    h=config['runtime']['h'],
                    samples=config['runtime']['samples'],
                    bin_path=bin_path
                )
                f.write(str(dp))

                # Run simulation with quaternion mode
                dp = runModel(
                    solver=solver,
                    integrator=integrator,
                    h=config['runtime']['h'],
                    samples=config['runtime']['samples'],
                    quaternion=True,
                    bin_path=bin_path
                )
                f.write(str(dp))

        print("Runtime experiments completed")

    # Stiffness experiments with multiprocessing
    if config['stiffness']['enabled']:
        print(f"Starting stiffness experiments with {config['stiffness']['threads']} threads")
        processes = []
        for i in range(config['stiffness']['threads']):
            # Create and start a process for each thread
            p = Process(
                target=stiffness,
                args=(
                    stiffness_dir,
                    f"stiffness{i}",
                    config['stiffness']['samples_per_thread'],
                    config['stiffness']['solver'],
                    config['stiffness']['integrator'],
                    config['stiffness']['h'],
                    bin_path
                )
            )
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in tqdm.tqdm(processes, desc="Running stiffness experiments"):
            p.join()
        print("Stiffness experiments completed")

    # Finite differences experiments
    if config['finite_differences']['enabled']:
        print("Starting finite differences experiments")
        output_file = f"{data_dir}/finite_differences.txt"
        print(f"Preparing to write finite differences results to {output_file}")
        with open(output_file, 'w') as f:
            j_start = config['finite_differences']['j_start']
            j_step = config['finite_differences']['j_step']
            j_end = config['finite_differences']['j_end']

            # Iterate over finite difference step sizes
            for j in tqdm.tqdm(list(np.arange(j_start, j_end + j_step/2, j_step)), desc="Running finite differences experiments"):
                j_val = float(f"{j}e-3")
                # Run simulation with rotation matrix
                dp = runModel(
                    solver=config['finite_differences']['solver'],
                    integrator=config['finite_differences']['integrator'],
                    h=config['finite_differences']['h'],
                    samples=config['finite_differences']['samples'],
                    fd_step=j_val,
                    bin_path=bin_path
                )
                f.write(str(dp))

                # Run simulation with quaternion mode
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

        print("Finite differences experiments completed")

    # Generate visualizations
    if config['visualizations']['enabled']:
        print("Generating visualizations")
        process_datafiles(
            stiffness_dir=stiffness_dir,
            data_dir=data_dir,
            graphs_dir=graphs_dir
        )
        print("Visualizations generated")

    # Pose calibration
    if config['pose_calibration']['enabled']:
        print("Starting pose calibration")
        # Download ground truth pose data if not present
        if not os.path.exists(GROUND_TRUTH_POSE) or not os.path.isfile(GROUND_TRUTH_POSE):
            print(f"Downloading ground truth pose data from {GROUND_TRUTH_POSE_URL}")
            urllib.request.urlretrieve(GROUND_TRUTH_POSE_URL, GROUND_TRUTH_POSE)
            print("Ground truth pose data downloaded")

        # Run pose calibration
        print(f"Running pose calibration with {config['pose_calibration']['threads']} threads")
        calibrate_pose(
            GROUND_TRUTH_POSE,
            data_dir,
            graphs_dir,
            config['pose_calibration']['threads']
        )
        print("Pose calibration completed")