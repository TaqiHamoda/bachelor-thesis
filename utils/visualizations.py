import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os

from .dataparser import fileParser


def meshGraph(data_x: np.ndarray, data_y: np.ndarray, data_z: np.ndarray, n: int = 100) -> None:
    """Creates a 2D color mesh plot from scattered data by averaging values in a grid.

    Args:
        data_x: X-coordinate data points.
        data_y: Y-coordinate data points.
        data_z: Z-value data points to be averaged.
        n: Number of grid points in each dimension (default: 100).
    """
    # Determine the range for x and y axes
    x_min, x_max = np.min(data_x), np.max(data_x)
    y_min, y_max = np.min(data_y), np.max(data_y)

    # Create a uniform grid
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    z = np.zeros((n, n, 2))  # Array to store summed z-values and counts

    # Bin data points into the grid and accumulate z-values
    for k in range(len(data_x)):
        d_x, d_y, d_z = data_x[k], data_y[k], data_z[k]
        i = round((n - 1) * (d_x - x_min) / (x_max - x_min))  # X index
        j = round((n - 1) * (d_y - y_min) / (y_max - y_min))  # Y index
        z[j, i, 0] += d_z  # Sum z-values
        z[j, i, 1] += 1   # Increment count

    # Compute averages for non-zero counts
    indx = z[:, :, 1].nonzero()
    z[:, :, 0][indx] /= z[:, :, 1][indx]

    # Set untouched grid points to None
    indx = np.where(z[:, :, 1] == 0)
    z[:, :, 0][indx] = None

    z = z[:, :, 0]  # Extract averaged z-values

    # Create color mesh plot with colorbar
    pcm = plt.pcolormesh(x, y, z, cmap="viridis")
    plt.colorbar(pcm)


def histogram(data: np.ndarray) -> None:
    """Generates a histogram with a complementary ECDF (empirical cumulative distribution function).

    Args:
        data: Data to plot in the histogram and ECDF.
    """
    fig, ax1 = plt.subplots()

    # Plot histogram on primary axis
    color1 = "chocolate"
    n, bins, patches = ax1.hist(data, bins="doane", rwidth=0.8, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Plot complementary ECDF on secondary axis
    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.ecdf(data, complementary=True, linestyle="--", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)


def forcesMeshGraphs(force1: np.ndarray, force2: np.ndarray, force3: np.ndarray, max_eigen: np.ndarray, 
                    s_iters: np.ndarray, f_calls: np.ndarray, p_err: np.ndarray, g_dir: str) -> None:
    """Creates color mesh plots for force components against various dependent variables.

    Args:
        force1: First force component (u_1).
        force2: Second force component (u_2).
        force3: Third force component (u_3).
        max_eigen: Maximum eigenvalue data.
        s_iters: Solver iterations data.
        f_calls: Function calls data.
        p_err: Position error data.
        g_dir: Directory to save the generated plots.
    """
    for d_variable in ("max eigen", "solver iterations", "function calls", "position error"):
        for g in ((1, 2), (1, 3), (2, 3)):
            # Select x and y data based on force component pair
            data_x = [force1, force2, force3][g[0] - 1]
            data_y = [force1, force2, force3][g[1] - 1]
            
            # Select dependent variable data
            data_z = {
                "max eigen": max_eigen,
                "solver iterations": s_iters,
                "function calls": f_calls,
                "position error": p_err
            }[d_variable]

            meshGraph(data_x, data_y, data_z, n=400)

            # Set axis labels with LaTeX formatting
            plt.xlabel(r"$u_{%s}$" % g[0])
            plt.ylabel(r"$u_{%s}$" % g[1])

            plt.tight_layout()
            plt.savefig(f"{g_dir}/{d_variable}_u_{g[0]}-{g[1]}.png")
            plt.close()


def betasMeshGraphs(beta1: np.ndarray, beta2: np.ndarray, beta3: np.ndarray, max_eigen: np.ndarray, 
                    s_iters: np.ndarray, f_calls: np.ndarray, p_err: np.ndarray, 
                    u_1: np.ndarray, u_2: np.ndarray, u_3: np.ndarray, g_dir: str) -> None:
    """Creates color mesh plots for beta parameters against various dependent variables.

    Args:
        beta1: First beta parameter.
        beta2: Second beta parameter.
        beta3: Third beta parameter.
        max_eigen: Maximum eigenvalue data.
        s_iters: Solver iterations data.
        f_calls: Function calls data.
        p_err: Position error data.
        u_1: First optimal control input.
        u_2: Second optimal control input.
        u_3: Third optimal control input.
        g_dir: Directory to save the generated plots.
    """
    for d_variable in ("max eigen", "solver iterations", "function calls", "position error", "u_1", "u_2", "u_3"):
        for g in ((1, 2), (1, 3), (2, 3)):
            # Select x and y data based on beta parameter pair
            data_x = [beta1, beta2, beta3][g[0] - 1]
            data_y = [beta1, beta2, beta3][g[1] - 1]
            
            # Select dependent variable data
            data_z = {
                "max eigen": max_eigen,
                "solver iterations": s_iters,
                "function calls": f_calls,
                "position error": p_err,
                "u_1": u_1,
                "u_2": u_2,
                "u_3": u_3
            }[d_variable]

            meshGraph(data_x, data_y, data_z, n=400)

            # Set axis labels with LaTeX formatting
            plt.xlabel(r"$\beta_{%s}$" % g[0])
            plt.ylabel(r"$\beta_{%s}$" % g[1])

            plt.tight_layout()
            plt.savefig(f"{g_dir}/{d_variable}_beta_{g[0]}-{g[1]}.png")
            plt.close()


def alphasMeshGraphs(alpha_2_3: np.ndarray, alpha_1_3: np.ndarray, max_eigen: np.ndarray, 
                     s_iters: np.ndarray, f_calls: np.ndarray, p_err: np.ndarray, 
                     u_1: np.ndarray, u_2: np.ndarray, u_3: np.ndarray, g_dir: str) -> None:
    """Creates color mesh plots for alpha differences against various dependent variables.

    Args:
        alpha_2_3: Differences between alpha_2 and alpha_3.
        alpha_1_3: Differences between alpha_1 and alpha_3.
        max_eigen: Maximum eigenvalue data.
        s_iters: Solver iterations data.
        f_calls: Function calls data.
        p_err: Position error data.
        u_1: First optimal control input.
        u_2: Second optimal control input.
        u_3: Third optimal control input.
        g_dir: Directory to save the generated plots.
    """
    for d_variable in ("max eigen", "solver iterations", "function calls", "position error", "u_1", "u_2", "u_3"):
        # Select dependent variable data
        data_z = {
            "max eigen": max_eigen,
            "solver iterations": s_iters,
            "function calls": f_calls,
            "position error": p_err,
            "u_1": u_1,
            "u_2": u_2,
            "u_3": u_3
        }[d_variable]

        meshGraph(alpha_2_3, alpha_1_3, data_z, n=400)

        # Set axis labels with LaTeX formatting
        plt.xlabel(r"$\alpha_2 - \alpha_3$")
        plt.ylabel(r"$\alpha_1 - \alpha_3$")

        # Customize ticks for angular differences
        locs = np.linspace(0, 2, 5) * np.pi
        labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
        plt.xticks(locs, labels)
        plt.yticks(locs, labels)

        plt.tight_layout()
        plt.savefig(f"{g_dir}/{d_variable}_Alpha 2 - Alpha 3-Alpha 1 - Alpha 3.png")
        plt.close()


def probabilityDensityGraphs(max_eigen: np.ndarray, s_iters: np.ndarray, f_calls: np.ndarray, 
                             p_err: np.ndarray, d_dir: str, g_dir: str) -> None:
    """Generates histograms and saves statistical summaries for dependent variables.

    Args:
        max_eigen: Maximum eigenvalue data.
        s_iters: Solver iterations data.
        f_calls: Function calls data.
        p_err: Position error data.
        d_dir: Directory to save statistical summary files.
        g_dir: Directory to save histogram plots.
    """
    for d_variable in ("max eigen", "solver iterations", "function calls", "position error"):
        # Select data for the variable
        data = {
            "max eigen": max_eigen,
            "solver iterations": s_iters,
            "function calls": f_calls,
            "position error": p_err
        }[d_variable]

        # Save statistical summary to file
        with open(f"{d_dir}/{d_variable}.txt", "w") as f:
            f.write(f"Mean: {np.average(data)}\n")
            f.write(f"Median: {np.median(data)}\n")
            f.write(f"STD: {np.std(data)}\n")
            f.write(f"Minimum: {np.min(data)}\n")
            f.write(f"Maximum: {np.max(data)}\n")

        histogram(data)

        plt.tight_layout()
        plt.savefig(f"{g_dir}/{d_variable}_histogram.png")
        plt.close()


def process_datafiles(stiffness_dir: str, data_dir: str, graphs_dir: str) -> None:
    """Processes data files to extract parameters and generate visualization plots.

    Args:
        stiffness_dir: Directory containing input data files.
        data_dir: Directory to save statistical summary files.
        graphs_dir: Directory to save visualization plots.
    """
    # Initialize lists to store extracted data
    alpha_1_3 = []
    alpha_2_3 = []
    beta_1 = []
    beta_2 = []
    beta_3 = []
    u_1 = []
    u_2 = []
    u_3 = []
    max_eigen = []
    s_iters = []
    f_calls = []
    p_err = []

    n_total = 0
    n_valid = 0
    for filename in os.listdir(stiffness_dir):
        file_path = f"{stiffness_dir}/{filename}"
        if not os.path.isfile(file_path):
            print(f"Skipping {filename}: not a valid file")
            continue

        print(f"Processing file: {filename}")
        # Parse data files into DataPoint objects
        datapoints = fileParser((filename,), stiffness_dir)
        print(f"Parsed {len(datapoints)} data points from {filename}")

        for dp in datapoints:
            n_total += 1
            if dp.s_iters == 100:  # Skip invalid data points
                print(f"Skipping invalid data point (s_iters=100) in {filename}")
                continue

            # Adjust alpha differences to ensure positive values
            alpha_1_3_val = dp.alphas[0] - dp.alphas[2] + 2*np.pi if dp.alphas[0] - dp.alphas[2] < 0 else dp.alphas[0] - dp.alphas[2]
            alpha_2_3_val = dp.alphas[1] - dp.alphas[2] + 2*np.pi if dp.alphas[1] - dp.alphas[2] < 0 else dp.alphas[1] - dp.alphas[2]
            
            alpha_1_3.append(alpha_1_3_val)
            alpha_2_3.append(alpha_2_3_val)
            beta_1.append(dp.betas[0])
            beta_2.append(dp.betas[1])
            beta_3.append(dp.betas[2])
            u_1.append(dp.u_optimal[0])
            u_2.append(dp.u_optimal[1])
            u_3.append(dp.u_optimal[2])
            max_eigen.append(dp.max_J_eigen)
            s_iters.append(dp.s_iters)
            f_calls.append(dp.f_calls)
            p_err.append(dp.pos_err)

            n_valid += 1

        datapoints.clear()
        print(f"Completed processing {filename}. Valid data points so far: {n_valid}/{n_total}")

    print(f"Total data points processed: {n_total}, Valid data points: {n_valid}")

    # Convert lists to numpy arrays for efficient processing
    alpha_1_3 = np.array(alpha_1_3)
    alpha_2_3 = np.array(alpha_2_3)
    beta_1 = np.array(beta_1)
    beta_2 = np.array(beta_2)
    beta_3 = np.array(beta_3)
    u_1 = np.array(u_1)
    u_2 = np.array(u_2)
    u_3 = np.array(u_3)
    max_eigen = np.array(max_eigen)
    s_iters = np.array(s_iters)
    f_calls = np.array(f_calls)
    p_err = np.array(p_err)
    print(f"Data conversion complete. Arrays created with {len(max_eigen)} valid entries")

    # Calculate and print probability of max_eigen > 20
    c_ecdf = scipy.stats.ecdf(max_eigen).sf
    probability = c_ecdf.evaluate(20) * 100
    print(f"Probability that max eigenvalue is greater than 20: {probability:.2f}%")

    # Generate visualization plots
    probabilityDensityGraphs(max_eigen, s_iters, f_calls, p_err, d_dir=data_dir, g_dir=graphs_dir)
    print("Probability density graphs completed")

    alphasMeshGraphs(alpha_2_3, alpha_1_3, max_eigen, s_iters, f_calls, p_err, u_1, u_2, u_3, g_dir=graphs_dir)
    print("Alpha mesh graphs completed")

    betasMeshGraphs(beta_1, beta_2, beta_3, max_eigen, s_iters, f_calls, p_err, u_1, u_2, u_3, g_dir=graphs_dir)
    print("Beta mesh graphs completed")

    forcesMeshGraphs(u_1, u_2, u_3, max_eigen, s_iters, f_calls, p_err, g_dir=graphs_dir)
    print("Forces mesh graphs completed")
