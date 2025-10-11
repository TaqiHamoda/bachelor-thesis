import json, time, os
from threading import Thread
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats

from .dataparser import runModel
from .visualizations import histogram

# Initial parameter offsets (in meters)
ORIGINAL_PARAMETERS = np.array([
    0.0021792646597059884, 0.01126373190601274, 0.0010055156330952485,
    0.0018443188994495565, -0.00018399665953102926, 0.0005040246029058141,
    -0.0005226400228246249, -0.004690433186634946, -0.0013207434413336808,
    -0.0035009435363276115, -0.006389302059844599, -0.0001766858857704401
])


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a rotation matrix.

    Reference: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    :param q: Quaternion as a numpy array [q_r, q_i, q_j, q_k].
    :return: 3x3 rotation matrix.
    """
    # Normalize the quaternion to handle non-unit quaternions
    q = q / np.linalg.norm(q)

    q_r, q_i, q_j, q_k = q

    rot = np.array([
        [1 - 2 * (q_j**2 + q_k**2), 2 * (q_i * q_j - q_k * q_r), 2 * (q_i * q_k + q_j * q_r)],
        [2 * (q_i * q_j + q_k * q_r), 1 - 2 * (q_i**2 + q_k**2), 2 * (q_j * q_k - q_i * q_r)],
        [2 * (q_i * q_k - q_j * q_r), 2 * (q_j * q_k + q_i * q_r), 1 - 2 * (q_i**2 + q_j**2)]
    ])

    return rot


def parse_dataset(filepath: str) -> List[np.ndarray]:
    """
    Parse the dataset from a CSV file.

    The CSV contains alpha and beta values, positions, and orientations for base, outer, middle, and inner segments.

    :param filepath: Path to the CSV file.
    :return: List of numpy arrays: [alphas, betas, pos_base, ori_base, pos_outer, ori_outer,
                                    pos_middle, ori_middle, pos_inner, ori_inner].
    """
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

    with open(filepath, "r") as f:
        line = f.readline()
        while line:
            parts = line.strip().split(",")

            alphas.append((float(parts[0]), float(parts[2]), float(parts[4])))
            betas.append((float(parts[1]) * 1e-3, float(parts[3]) * 1e-3, float(parts[5]) * 1e-3))

            pos_base.append((float(parts[12]) * 1e-3, float(parts[13]) * 1e-3, float(parts[14]) * 1e-3))
            ori_base.append((float(parts[15]), float(parts[16]), float(parts[17]), float(parts[18])))

            pos_outer.append((float(parts[19]) * 1e-3, float(parts[20]) * 1e-3, float(parts[21]) * 1e-3))
            ori_outer.append((float(parts[22]), float(parts[23]), float(parts[24]), float(parts[25])))

            pos_middle.append((float(parts[26]) * 1e-3, float(parts[27]) * 1e-3, float(parts[28]) * 1e-3))
            ori_middle.append((float(parts[29]), float(parts[30]), float(parts[31]), float(parts[32])))

            pos_inner.append((float(parts[33]) * 1e-3, float(parts[34]) * 1e-3, float(parts[35]) * 1e-3))
            ori_inner.append((float(parts[36]), float(parts[37]), float(parts[38]), float(parts[39])))

            line = f.readline()

    return [
        np.array(alphas), np.array(betas), np.array(pos_base), np.array(ori_base),
        np.array(pos_outer), np.array(ori_outer), np.array(pos_middle), np.array(ori_middle),
        np.array(pos_inner), np.array(ori_inner)
    ]


def compute_tips(alphas: np.ndarray, betas: np.ndarray, start: int, end: int, tips: np.ndarray) -> None:
    """
    Compute tip positions for a range of data points using the runModel function.

    This function is designed to be run in a thread for parallel processing.

    :param alphas: Array of alpha values.
    :param betas: Array of beta values.
    :param start: Starting index for computation.
    :param end: Ending index for computation.
    :param tips: Pre-allocated array to store tip positions.
    """
    for k in range(start, end):
        dp = runModel(
            "Broyden", "AB2", 1e-3, quaternion=True, without_check=True,
            alphas=alphas[k], betas=betas[k], forces=[0, 0, 0]
        )
        tips[k] = dp.tip_positions


def construct_tips(alphas: np.ndarray, betas: np.ndarray, thread_count: int) -> np.ndarray:
    """
    Construct tip positions using multi-threading for all data points.

    :param alphas: Array of alpha values.
    :param betas: Array of beta values.
    :return: Array of tip positions (n, 3, 3).
    """
    num_points = alphas.size
    tips = np.zeros((num_points, 3, 3))
    threads: List[Thread] = []

    step = num_points // thread_count
    for h in range(0, num_points, step):
        thread_end = min(h + step, num_points)
        thread = Thread(target=compute_tips, args=(alphas, betas, h, thread_end, tips))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return tips


def compute_position_errors(
    parameters: np.ndarray,
    data: List[np.ndarray],
    tips: np.ndarray
) -> np.ndarray:
    """
    Compute position errors for all segments relative to model tips.

    :param parameters: Optimization parameters (deltas for positions).
    :param data: Parsed dataset.
    :param tips: Model tip positions.
    :return: Position errors for inner segment (n, 3).
    """
    num_points = data[0].size
    pos_err = np.zeros((num_points, 3))

    delta_pos_base = parameters[0:3]
    delta_pos_outer = parameters[3:6]
    delta_pos_middle = parameters[6:9]
    delta_pos_inner = parameters[9:12]

    _, _, pos_base, ori_base, pos_outer, ori_outer, pos_middle, ori_middle, pos_inner, ori_inner = data

    for i in range(num_points):
        t_aurora_base = np.eye(4)
        t_aurora_outer = np.eye(4)
        t_aurora_middle = np.eye(4)
        t_aurora_inner = np.eye(4)

        t_aurora_base[:3, :3] = quaternion_to_rotation(ori_base[i])
        t_aurora_base[:3, 3] = pos_base[i] + delta_pos_base

        t_aurora_outer[:3, :3] = quaternion_to_rotation(ori_outer[i])
        t_aurora_outer[:3, 3] = pos_outer[i] + delta_pos_outer

        t_aurora_middle[:3, :3] = quaternion_to_rotation(ori_middle[i])
        t_aurora_middle[:3, 3] = pos_middle[i] + delta_pos_middle

        t_aurora_inner[:3, :3] = quaternion_to_rotation(ori_inner[i])
        t_aurora_inner[:3, 3] = pos_inner[i] + delta_pos_inner

        t_base_aurora = np.linalg.inv(t_aurora_base)
        t_base_inner = t_base_aurora @ t_aurora_inner

        pos_err[i] = t_base_inner[:3, 3] - tips[i][0]  # Only inner for outlier detection

    return pos_err


def residuals(
    parameters: np.ndarray,
    data: List[np.ndarray],
    tips: np.ndarray
) -> np.ndarray:
    """
    Compute residuals for least squares optimization.

    Residuals are the position differences (in mm) for all components and segments.
    This allows minimizing the sum of squared errors (MSE) in position.

    :param parameters: Optimization parameters (deltas for positions).
    :param data: Parsed dataset (subset for optimization).
    :param tips: Model tip positions (subset).
    :return: Flat array of residuals (n * 3 segments * 3 coords).
    """
    num_points = data[0].size

    residuals_list: List[float] = []

    delta_pos_base = parameters[0:3]
    delta_pos_outer = parameters[3:6]
    delta_pos_middle = parameters[6:9]
    delta_pos_inner = parameters[9:12]

    _, _, pos_base, ori_base, pos_outer, ori_outer, pos_middle, ori_middle, pos_inner, ori_inner = data

    for i in range(num_points):
        t_aurora_base = np.eye(4)
        t_aurora_outer = np.eye(4)
        t_aurora_middle = np.eye(4)
        t_aurora_inner = np.eye(4)

        t_aurora_base[:3, :3] = quaternion_to_rotation(ori_base[i])
        t_aurora_base[:3, 3] = pos_base[i] + delta_pos_base

        t_aurora_outer[:3, :3] = quaternion_to_rotation(ori_outer[i])
        t_aurora_outer[:3, 3] = pos_outer[i] + delta_pos_outer

        t_aurora_middle[:3, :3] = quaternion_to_rotation(ori_middle[i])
        t_aurora_middle[:3, 3] = pos_middle[i] + delta_pos_middle

        t_aurora_inner[:3, :3] = quaternion_to_rotation(ori_inner[i])
        t_aurora_inner[:3, 3] = pos_inner[i] + delta_pos_inner

        t_base_aurora = np.linalg.inv(t_aurora_base)
        t_base_outer = t_base_aurora @ t_aurora_outer
        t_base_middle = t_base_aurora @ t_aurora_middle
        t_base_inner = t_base_aurora @ t_aurora_inner

        # Append differences in mm for each component
        for diff in (t_base_outer[:3, 3] - tips[i][2]):
            residuals_list.append(1e3 * diff)
        for diff in (t_base_middle[:3, 3] - tips[i][1]):
            residuals_list.append(1e3 * diff)
        for diff in (t_base_inner[:3, 3] - tips[i][0]):
            residuals_list.append(1e3 * diff)

    return np.array(residuals_list)


def save_data(parameters: np.ndarray, pos_err: np.ndarray, function_calls: int, data_dir: str, graphs_dir: str) -> None:
    """
    Save optimization results to JSON and generate a histogram of position errors.

    :param parameters: Optimized parameters.
    :param pos_err: Position errors (in meters).
    :param function_calls: Number of function calls during optimization.
    """
    p_err_mm = 1e3 * np.linalg.norm(pos_err, axis=1)

    with open(f"{data_dir}/least_squares.json", "w") as f:
        json.dump({
            "Function Calls": function_calls,
            "Optimized parameters": parameters.tolist(),
            "Average Position Error (mm)": np.average(p_err_mm),
            "Median Position Error (mm)": np.median(p_err_mm),
            "STD Position Error (mm)": np.std(p_err_mm),
            "Min Position Error (mm)": np.min(p_err_mm),
            "Max Position Error (mm)": np.max(p_err_mm),
            "Position Errors (mm)": p_err_mm.tolist(),
        }, f)

    histogram(p_err_mm)
    plt.tight_layout()
    plt.savefig(f"{graphs_dir}/dataset_histogram.png")
    plt.close()


def get_indices_without_outliers(pos_err: np.ndarray) -> np.ndarray:
    """
    Identify indices without outliers using IQR method on each axis.

    :param pos_err: Position errors (DATA_N, 3).
    :return: Array of indices without outliers.
    """
    indices = np.ones(pos_err.size, dtype=bool)
    for a in range(3):  # Check each axis
        errors = pos_err[:, a]
        Q1 = np.percentile(errors, 25, method='midpoint')
        Q3 = np.percentile(errors, 75, method='midpoint')
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        indices &= (errors > lower_limit) & (errors < upper_limit)

    return np.arange(pos_err.size)[indices]


def optimize_parameters(
    initial_params: np.ndarray,
    data: List[np.ndarray],
    tips: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Optimize parameters using least squares.

    :param initial_params: Initial guess for parameters.
    :param data: Subset of parsed dataset for optimization.
    :param tips: Subset of tip positions.
    :return: Optimized parameters and number of function calls.
    """
    num_points = data[0].size

    print("Optimization started")

    function_calls = 0

    def residuals_wrapper(params):
        function_calls += 1
        return residuals(params, data, tips, num_points)

    start = time.perf_counter()
    result = scipy.optimize.least_squares(
        residuals_wrapper, initial_params, method="trf", jac="3-point", verbose=2
    )
    end = time.perf_counter()

    print(f"Optimization runtime: {end - start} seconds")
    return result.x, function_calls


def calibrate_pose(ground_truth_path, data_dir, graphs_dir, thread_count):
    indices_file_path = f"{data_dir}/indices.json"

    # Parse the full dataset
    data = parse_dataset(ground_truth_path)
    print("Parsed dataset")

    alphas, betas = data[0], data[1]

    # Compute tips for all data points
    all_tips = construct_tips(alphas, betas, thread_count)

    if not os.path.exists(indices_file_path) or not os.path.isfile(indices_file_path):
        # Compute position errors with original parameters for outlier detection
        pos_err = compute_position_errors(ORIGINAL_PARAMETERS, data, all_tips)
        with open(indices_file_path, "w") as f:
            json.dump(pos_err.tolist(), f)
    else:
        with open(indices_file_path, "r") as f:
            pos_err = np.array(json.load(f))

    # Remove outliers
    valid_indices = get_indices_without_outliers(pos_err)
    print(f"Number of valid indices after outlier removal: {len(valid_indices)}")

    # Shuffle and select N samples
    np.random.shuffle(valid_indices)

    # Subset data and tips
    subset_data = [array[valid_indices] for array in data]
    subset_tips = all_tips[valid_indices]

    # Optimize
    optimized_params, func_calls = optimize_parameters(ORIGINAL_PARAMETERS, subset_data, subset_tips)

    # Compute final position errors (only inner for stats)
    final_pos_err = compute_position_errors(optimized_params, subset_data, subset_tips)

    # Save final data
    save_data(optimized_params, final_pos_err, func_calls, data_dir, graphs_dir)

    # Compute and print ECDF statistic
    with open(f"{data_dir}/least_squares.json", 'r') as f:
        results = json.load(f)
        p_err_mm = results["Position Errors (mm)"]
        ecdf = scipy.stats.ecdf(p_err_mm).cdf
        print(f"Probability that values are less than 6.3 mm: {ecdf.evaluate(6.3) * 100}%")  # Should be around 92.67%