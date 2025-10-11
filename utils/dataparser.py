import subprocess


class DataPoint:
    """Represents a single data point for a simulation run, storing function call parameters and performance results."""
    
    def __init__(self) -> None:
        # Function Call Data: Parameters for the simulation
        self.solver: str = ""  # Solver type used in the simulation
        self.integrator: str = ""  # Integrator type used in the simulation
        self.step_size: float = 0  # Step size for numerical integration
        self.without_check: bool = False  # Flag to disable checks
        self.quaternion_mode: bool = False  # Flag for quaternion mode
        self.fd_step: float = 0  # Finite difference step size
        self.num_samples: int = 0  # Number of samples in the simulation
        self.e: list[float] = []  # List of energy values (kbt)
        self.kappas: list[float] = []  # List of kappa values

        # Performance Results: Metrics from the simulation
        self.runtime: float = 0  # Runtime in microseconds
        self.res_err: float = 0  # Residual error
        self.pos_err: float = 0  # Position error in millimeters
        self.ori_err: float = 0  # Orientation error in degrees
        self.s_iters: float = 0  # Number of solver iterations
        self.f_calls: float = 0  # Number of function calls

        # Model Data: Simulation outputs
        self.alphas: list[float] = []  # Alpha parameters
        self.betas: list[float] = []  # Beta parameters
        self.u_guess: list[float] = []  # Initial guess for control inputs
        self.u_optimal: list[float] = []  # Optimal control inputs
        self.tip_positions: list[list[float]] = []  # Tip positions for each tube
        self.max_J_eigen: float = 0  # Maximum absolute eigenvalue of Jacobian

    def __str__(self) -> str:
        """Returns a string representation of the DataPoint, including command-line arguments and results."""
        # Construct command-line string
        s = f"./bin -s {self.solver} -i {self.integrator} -h {self.step_size}"
        if self.without_check:
            s += " -w"
        if self.quaternion_mode:
            s += " -q"
        if self.fd_step != 0:
            s += f" -j {self.fd_step}"
        if self.num_samples > 1:
            s += f" -n {self.num_samples}"
        if self.e:
            s += f" -e {' '.join([str(kbt) for kbt in self.e])}"
        if self.kappas:
            s += f" -k {' '.join([str(kappa) for kappa in self.kappas])}"
        s += "\n"

        # Append performance metrics
        s += f"Runtime: {self.runtime} microseconds\n"
        s += f"Residual Error: {self.res_err}\n"
        s += f"Position Error: {self.pos_err} millimetres\n"
        s += f"Orientation Error: {self.ori_err} degrees\n"
        s += f"Solver Iterations: {self.s_iters}\n"
        s += f"Function Calls: {self.f_calls}\n"

        # Append model data for single-sample runs
        if self.num_samples == 1:
            s += f"Alphas: {' '.join([str(i) for i in self.alphas])}\n"
            s += f"Betas: {' '.join([str(i) for i in self.betas])}\n"
            s += f"U Guess: {' '.join([str(i) for i in self.u_guess])}\n"
            s += f"U Optimal: {' '.join([str(i) for i in self.u_optimal])}\n"
            for t, pos in enumerate(self.tip_positions, 1):
                s += f"Tip Position of Tube{t} (meters): {' '.join([str(i) for i in pos])}\n"
            s += f"Max Absolute Eigenvalue of A: {self.max_J_eigen}\n"

        return s + "\n"


def extractFunctionCallData(fc_data: list[str], dp: DataPoint) -> None:
    """Parses command-line arguments and populates the DataPoint object with function call data.

    Args:
        fc_data: List of command-line argument strings.
        dp: DataPoint object to populate with parsed data.
    """
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
    """Extracts values from a result string by splitting on ': ' and removing whitespace.

    Args:
        data: String containing result data in the format 'Label: value'.

    Returns:
        List of values split by spaces.
    """
    d = data.split(": ")
    return d[1].strip().split(" ")


def convertToArray(data: list[str]) -> list[float]:
    """Converts a list of strings to a list of floats, ignoring invalid entries.

    Args:
        data: List of strings to convert.

    Returns:
        List of floats parsed from valid string entries.
    """
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
    """Parses a string containing simulation data into a DataPoint object.

    Args:
        datapoint: String containing simulation data with command-line arguments and results.

    Returns:
        Populated DataPoint object.
    """
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
            dp.runtime = float(getResultValue(l)[0])
        elif data[0] == "Residual":
            dp.res_err = float(getResultValue(l)[0])
        elif data[0] == "Position":
            dp.pos_err = float(getResultValue(l)[0])
        elif data[0] == "Orientation":
            dp.ori_err = float(getResultValue(l)[0])
        elif data[0] == "Solver":
            dp.s_iters = float(getResultValue(l)[0])
        elif data[0] == "Function":
            dp.f_calls = float(getResultValue(l)[0])
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


def fileParser(filenames: list[str], data_dir: str) -> list[DataPoint]:
    """Parses multiple files containing simulation data into a list of DataPoint objects.

    Args:
        filenames: List of filenames to parse.
        data_dir: Directory containing the data files.

    Returns:
        List of DataPoint objects parsed from the files.
    """
    datapoints = []
    for filename in filenames:
        with open(f"{data_dir}/{filename}", "r") as f:
            datapoint = ""
            for l in f:
                datapoint += l
                if l == "\n":
                    dp = dataParser(datapoint)
                    if dp.solver != "":
                        datapoints.append(dp)
                        datapoint = ""
    return datapoints


def runModel(solver: str, integrator: str, h: float, bin_path: str = "./bin", fd_step: float=0, without_check: bool = False, quaternion: bool=False, samples: int=1, alphas: list[float]=[], betas: list[float]=[], forces: list[float]=[], e: list[float]=[], kappas: list[float]=[]) -> DataPoint:
    """Runs a simulation model with specified parameters and returns parsed results as a DataPoint.

    Args:
        solver: Solver type to use.
        integrator: Integrator type to use.
        h: Step size for numerical integration.
        bin_path: Path to the executable binary (default: './bin').
        fd_step: Finite difference step size (default: 0).
        without_check: Flag to disable checks (default: False).
        quaternion: Flag to enable quaternion mode (default: False).
        samples: Number of samples to run (default: 1).
        alphas: List of alpha parameters (default: []).
        betas: List of beta parameters (default: []).
        forces: List of force values (default: []).
        e: List of energy values (kbt) (default: []).
        kappas: List of kappa values (default: []).

    Returns:
        DataPoint object containing the parsed simulation results.
    """
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