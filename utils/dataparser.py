import subprocess


class DataPoint:
    def __init__(self) -> None:
        # Function Call Data
        self.solver = ""
        self.integrator = ""
        self.step_size = 0
        self.without_check = False
        self.quaternion_mode = False
        self.fd_step = 0
        self.num_samples = 0
        self.e = []
        self.kappas = []

        # Performance Results
        self.runtime = 0
        self.res_err = 0
        self.pos_err = 0
        self.ori_err = 0
        self.s_iters = 0
        self.f_calls = 0

        # Model Data
        self.alphas = []
        self.betas = []
        self.u_guess = []
        self.u_optimal = []
        self.tip_positions = []
        self.max_J_eigen = 0

    def __str__(self) -> str:
        s = f"./bin -s {self.solver} -i {self.integrator} -h {self.step_size}"

        if self.without_check:
            s += " -w"

        if self.quaternion_mode:
            s += " -q"

        if self.fd_step != 0:
            s += f" -j {self.fd_step}"

        if self.num_samples > 1:
            s += f" -n {self.num_samples}"

        if self.e != []:
            s += f" -e {' '.join([str(kbt) for kbt in self.e])}"

        if self.kappas != []:
            s += f" -k {' '.join([str(kappa) for kappa in self.kappas])}"

        s += "\n"

        s += f"Runtime: {self.runtime} microseconds\n"
        s += f"Residual Error: {self.res_err}\n"
        s += f"Position Error: {self.pos_err} millimetres\n"
        s += f"Orientation Error: {self.ori_err} degrees\n"
        s += f"Solver Iterations: {self.s_iters}\n"
        s += f"Function Calls: {self.f_calls}\n"

        if self.num_samples == 1:
            s += f"Alphas: " + " ".join([str(i) for i in self.alphas]) + "\n"
            s += f"Betas: " + " ".join([str(i) for i in self.betas]) + "\n"
            s += f"U Guess: " + " ".join([str(i) for i in self.u_guess]) + "\n"
            s += f"U Optimal: " + " ".join([str(i) for i in self.u_optimal]) + "\n"

            for t in range(len(self.tip_positions)):
                s += f"Tip Position of Tube{t + 1} (meters): " + " ".join([str(i) for i in self.tip_positions[t]]) + "\n"

            s += f"Max Absolute Eigenvalue of A: {self.max_J_eigen}\n"

        return s + "\n"


def extractFunctionCallData(fc_data: list[str], dp: DataPoint) -> None:
    i = 1
    while i < len(fc_data):
        if fc_data[i] == "-s":
            dp.solver = fc_data[i + 1]
            i += 2
        elif fc_data[i] == "-i":
            dp.integrator = fc_data[i + 1]
            i += 2
        elif fc_data[i] == "-h":
            dp.step_size = float(fc_data[i + 1])
            i += 2
        elif fc_data[i] == "-j":
            dp.fd_step = float(fc_data[i + 1])
            i += 2
        elif fc_data[i] == "-e":
            dp.e = convertToArray(fc_data[i + 1: i + 7])
            i += 7
        elif fc_data[i] == "-k":
            dp.kappas = convertToArray(fc_data[i + 1: i + 4])
            i += 4
        elif fc_data[i] == "-n":
            dp.num_samples = int(fc_data[i + 1])
            i += 2
        elif fc_data[i] == "-q":
            dp.quaternion_mode = True
            i += 1
        elif fc_data[i] == "-w":
            dp.without_check = True
            i += 1
        else:
            i += 1


def getResultValue(data: str) -> list[str]:
    d = data.split(": ")
    return d[1].strip().split(" ")


def convertToArray(data: list[str]) -> list[float]:
    result = []

    for d in data:
        if d == "":
            continue

        try:
            result.append(float(d))
        except Exception as e:
            print(e)
            continue

    return result


def dataParser(datapoint: str) -> DataPoint:
    dp = DataPoint()
    for l in datapoint.split("\n"):
        data = l.strip().split(" ")
        if len(data) == 1:
            if dp.num_samples == 0:
                dp.num_samples = 1
            break
        elif data[0] == "./bin":
            extractFunctionCallData(data, dp)
        elif data[0] == "Runtime:":
            dp.runtime = int(getResultValue(l)[0])
        elif data[0] == "Residual":
            dp.res_err = float(getResultValue(l)[0])
        elif data[0] == "Position":
            dp.pos_err = float(getResultValue(l)[0])
        elif data[0] == "Orientation":
            dp.ori_err = float(getResultValue(l)[0])
        elif data[0] == "Solver":
            dp.s_iters = int(getResultValue(l)[0])
        elif data[0] == "Function":
            dp.f_calls = int(getResultValue(l)[0])
        elif data[0] == "Alphas:":
            dp.alphas = convertToArray(getResultValue(l))
        elif data[0] == "Betas:":
            dp.betas = convertToArray(getResultValue(l))
        elif data[0] == "U" and data[1] == "Guess:":
            dp.u_guess = convertToArray(getResultValue(l))
        elif data[0] == "U" and data[1] == "Optimal:":
            dp.u_optimal = convertToArray(getResultValue(l))
        elif data[0] == "Tip":
            dp.tip_positions.append(convertToArray(getResultValue(l)))
        elif data[0] == "Max":
            dp.max_J_eigen = float(getResultValue(l)[0])

    return dp


def fileParser(filenames: list[str], dir="./data/") -> list[DataPoint]:
    datapoints = []
    
    for filename in filenames:
        with open(f"{dir}/{filename}", "r") as f:
            datapoint = ""
            for l in f:
                datapoint += l

                if l == "\n":
                    dp = dataParser(datapoint)

                    if dp.solver != "":  # If dp is initialized
                        datapoints.append(dp)
                        datapoint = ""

    return datapoints


def runModel(solver: str, integrator: str, h: float, bin_path: str = "./bin", fd_step: float=0, without_check: bool = False, quaternion: bool=False, samples: int=1, alphas: list[float]=[], betas: list[float]=[], forces: list[float]=[], e: list[float]=[], kappas: list[float]=[]) -> DataPoint:
    args = [bin_path, "-s", solver, "-i", integrator, "-h", str(h)]

    if fd_step > 0:
        args += ["-j", str(fd_step)]

    if without_check:
        args += ["-w"]

    if quaternion:
        args += ["-q"]

    if samples > 1:
        args += ["-n", str(samples)]

    if len(alphas) != 0:
        args += ["-a"] + [str(alpha) for alpha in alphas]

    if len(betas) != 0:
        args += ["-b"] + [str(beta) for beta in betas]

    if len(forces) != 0:
        args += ["-f"] + [str(force) for force in forces]

    if len(e) != 0:
        args += ["-e"] + [str(kbt) for kbt in e]

    if len(kappas) != 0:
        args += ["-k"] + [str(kappa) for kappa in kappas]

    datapoint = "./bin " + " ".join(args[1:]) + "\n" + subprocess.check_output(args).decode()
    return dataParser(datapoint)