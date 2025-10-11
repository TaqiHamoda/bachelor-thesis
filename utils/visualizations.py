import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os

from .dataparser import fileParser


def meshGraph(data_x, data_y, data_z, n=100):
    x_min, x_max = np.min(data_x), np.max(data_x)
    y_min, y_max = np.min(data_y), np.max(data_y)

    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    z = np.zeros((n, n, 2))

    for k in range(len(data_x)):
        d_x, d_y, d_z = data_x[k], data_y[k], data_z[k]

        i = round((n - 1)*(d_x - x_min)/(x_max - x_min))
        j = round((n - 1)*(d_y - y_min)/(y_max - y_min))

        z[j, i, 0] += d_z
        z[j, i, 1] += 1

    # Get the averages
    indx = z[:, :, 1].nonzero()
    z[:, :, 0][indx] /= z[:, :, 1][indx]

    # Overwrite untouched areas with None
    indx = np.where(z[:, :, 1] == 0)
    z[:, :, 0][indx] = None

    z = z[:, :, 0]

    pcm = plt.pcolormesh(x, y, z, cmap="viridis")
    c_bar = plt.colorbar(pcm)


def histogram(data: np.ndarray):
    # Plot histogram
    fig, ax1 = plt.subplots()

    color1 = "chocolate"
    n, bins, patches = ax1.hist(data, bins="doane", rwidth=0.8, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()

    color2 = "tab:blue"
    ax2.ecdf(data, complementary=True, linestyle="--", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)


def forcesMeshGraphs(force1, force2, force3, max_eigen, s_iters, f_calls, p_err, g_dir):
    # Color Mesh Graphs
    for d_variable in ("max eigen", "solver iterations", "function calls", "position error"):
        for g in ((1, 2), (1, 3), (2, 3)):
            if g[0] == 1:
                data_x = force1
            elif g[0] == 2:
                data_x = force2
            elif g[0] == 3:
                data_x = force3
            
            if g[1] == 1:
                data_y = force1
            elif g[1] == 2:
                data_y = force2
            elif g[1] == 3:
                data_y = force3
            
            if d_variable == "max eigen":
                data_z = max_eigen
            elif d_variable == "solver iterations":
                data_z = s_iters
            elif d_variable == "function calls":
                data_z = f_calls
            elif d_variable == "position error":
                data_z = p_err

            meshGraph(data_x, data_y, data_z, n=400)

            plt.xlabel(r"$u_{%s}$" % (g[0]))
            plt.ylabel(r"$u_{%s}$" % (g[1]))

            plt.tight_layout()
            plt.savefig(f"{g_dir}/{d_variable}_u_{g[0]}-{g[1]}.png")
            plt.close()


def betasMeshGraphs(beta1, beta2, beta3, max_eigen, s_iters, f_calls, p_err, u_1, u_2, u_3, g_dir):
    # Color Mesh Graphs
    for d_variable in ("max eigen", "solver iterations", "function calls", "position error", "u_1", "u_2", "u_3"):
        for g in ((1, 2), (1, 3), (2, 3)):
            if g[0] == 1:
                data_x = beta1
            elif g[0] == 2:
                data_x = beta2
            elif g[0] == 3:
                data_x = beta3
            
            if g[1] == 1:
                data_y = beta1
            elif g[1] == 2:
                data_y = beta2
            elif g[1] == 3:
                data_y = beta3
            
            if d_variable == "max eigen":
                data_z = max_eigen
            elif d_variable == "solver iterations":
                data_z = s_iters
            elif d_variable == "function calls":
                data_z = f_calls
            elif d_variable == "position error":
                data_z = p_err
            elif d_variable == "u_1":
                data_z = u_1
            elif d_variable == "u_2":
                data_z = u_2
            elif d_variable == "u_3":
                data_z = u_3

            meshGraph(data_x, data_y, data_z, n=400)

            plt.xlabel(r"$\beta_{%s}$" % (g[0]))
            plt.ylabel(r"$\beta_{%s}$" % (g[1]))

            plt.tight_layout()
            plt.savefig(f"{g_dir}/{d_variable}_beta_{g[0]}-{g[1]}.png")
            plt.close()


def alphasMeshGraphs(alpha_2_3, alpha_1_3, max_eigen, s_iters, f_calls, p_err, u_1, u_2, u_3, g_dir):
    for d_variable in ("max eigen", "solver iterations", "function calls", "position error", "u_1", "u_2", "u_3"):
        if d_variable == "max eigen":
            data_z = max_eigen
        elif d_variable == "solver iterations":
            data_z = s_iters
        elif d_variable == "function calls":
            data_z = f_calls
        elif d_variable == "position error":
            data_z = p_err
        elif d_variable == "u_1":
            data_z = u_1
        elif d_variable == "u_2":
            data_z = u_2
        elif d_variable == "u_3":
            data_z = u_3

        meshGraph(alpha_2_3, alpha_1_3, data_z, n=400)

        plt.xlabel(r"$\alpha_2$ - $\alpha_3$")
        plt.ylabel(r"$\alpha_1$ - $\alpha_3$")

        locs = np.linspace(0, 2, 5)*np.pi
        labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
        plt.xticks(locs, labels)
        plt.yticks(locs, labels)

        plt.tight_layout()
        plt.savefig(f"{g_dir}/{d_variable}_Alpha 2 - Alpha 3-Alpha 1 - Alpha 3.png")
        plt.close()


def probabilityDensityGraphs(max_eigen, s_iters, f_calls, p_err, d_dir, g_dir):
    for d_variable in ("max eigen", "solver iterations", "function calls", "position error"):
        if d_variable == "max eigen":
            data = max_eigen
        elif d_variable == "solver iterations":
            data = s_iters
        elif d_variable == "function calls":
            data = f_calls
        elif d_variable == "position error":
            data = p_err

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


def process_datafiles(stiffness_dir, data_dir, graphs_dir):
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
        if not os.path.isfile(f"{stiffness_dir}/{filename}"):
            continue

        print(f"Processing {filename}")

        datapoints = fileParser((filename, ))
        for dp in datapoints:
            n_total += 1
            if dp.s_iters == 100:
                continue

            alpha_1_3.append(dp.alphas[0] - dp.alphas[2] + 2*np.pi if dp.alphas[0] - dp.alphas[2] < 0 else dp.alphas[0] - dp.alphas[2])
            alpha_2_3.append(dp.alphas[1] - dp.alphas[2] + 2*np.pi if dp.alphas[1] - dp.alphas[2] < 0 else dp.alphas[1] - dp.alphas[2])

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

    print(f"{n_total} Datapoints Processed")
    print(f"{n_valid} Valid Datapoints")

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

    c_ecdf = scipy.stats.ecdf(max_eigen).sf
    print(f"Probability that values are greater than 20: {c_ecdf.evaluate(20) * 100}%")  # Should be around 0.0010612224752293287%

    probabilityDensityGraphs(max_eigen, s_iters, f_calls, p_err, d_dir=data_dir, g_dir=graphs_dir)
    alphasMeshGraphs(alpha_2_3, alpha_1_3, max_eigen, s_iters, f_calls, p_err, u_1, u_2, u_3, g_dir=graphs_dir)
    betasMeshGraphs(beta_1, beta_2, beta_3, max_eigen, s_iters, f_calls, p_err, u_1, u_2, u_3, g_dir=graphs_dir)
    forcesMeshGraphs(u_1, u_2, u_3, max_eigen, s_iters, f_calls, p_err, g_dir=graphs_dir)