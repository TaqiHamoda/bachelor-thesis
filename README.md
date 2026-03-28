# From Milliseconds to Microseconds: Optimizing the Static Forward Kinematics of CTCRs

[![DOI](https://zenodo.org/badge/1073692035.svg)](https://doi.org/10.5281/zenodo.19283166)

This repository contains the C++ implementation along with my Python legacy code for the report **"From Milliseconds to Microseconds: Optimizing the Static Forward Kinematics of CTCRs for use in Real-Time Systems"**. This research was conducted as part of my bachelor thesis at the **Continuum Robotics Laboratory, University of Toronto** and was entirely funded by the Natural Sciences and Engineering Research Council of Canada (**NSERC: [RGPIN-2019-04846]**).

The project focuses on dramatically improving the computational performance of the static Cosserat rod model for Concentric Tube Continuum Robots (CTCRs). By applying a series of numerical optimizations, we were able to reduce the forward kinematics computation time to just **120 microseconds**—a reduction of up to **90%** in function calls compared to standard methods, without sacrificing accuracy.

## 📜 A Note on the Publication Status

You might notice that the accompanying report was not formally published. I believe in transparency in research, and I want to share the story behind this work.

During my thesis, we achieved some very exciting results, particularly in runtime optimization. However, a cornerstone of our stiffness analysis relied on a statistical approach to approximate the upper bound of the system's eigenvalues. While this method proved effective for our specific CTCR parameters and generated consistent results, it lacked the rigorous, definite mathematical proof that is rightfully expected for publications in top-tier robotics journals. We had strong empirical evidence, but not a formal, analytical guarantee of the system's non-stiffness.

Additionally, our exploration led to an interesting possible discovery: a potential numerical proxy for assessing the robot's elastic stability and predicting "snapping" behavior. This was a promising avenue, but due to the time constraints of a bachelor's thesis, I couldn't explore it to the depth it deserved.

Despite not crossing the publication finish line, I am incredibly proud of this work and the collaboration that made it possible. I'm sharing it with the hope that it can be a useful resource for other students, researchers, and engineers working on continuum robotics.

## 🚀 Getting Started

To get the model running on your local machine, follow these simple steps.

### Prerequisites

The code relies on the **Eigen3** library for linear algebra operations. Most Linux distributions offer it through their package manager.

  * **On Ubuntu/Debian:**
    ```sh
    sudo apt-get install libeigen3-dev
    ```
  * **On Fedora/CentOS:**
    ```sh
    sudo dnf install eigen3-devel
    ```

### Building the Project

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/TaqiHamoda/bachelor-thesis.git
    cd bachelor-thesis
    ```

2.  **Compile the source files using the `Makefile`:**
    The provided `Makefile` handles the entire build process.

      * **For a release build (optimized for performance):**
        This is the default mode. It uses the `-O3` optimization flag and creates an executable named `bin`.

        ```sh
        make
        ```

      * **For a debug build:**
        This mode is useful for development and debugging. It uses the `-g` flag for debugging symbols and creates an executable named `bin.debug`.

        ```sh
        make mode=debug
        ```

### Running an Experiment

The executable is controlled via command-line arguments. The solver, integrator, and step size are required.

```sh
./bin -s <solver> -i <integrator> -h <step_size> [options]
```

**Required Arguments:**

  * `-s`: The non-linear solver to use.
      * `Newton`: Newton-Raphson method.
      * `Broyden`: Broyden's method (secant update).
  * `-i`: The numerical integrator for the ODE system.
      * `ForwardEuler`
      * `RK2`, `RK3`, `RK4` (Runge-Kutta methods of orders 2, 3, and 4).
      * `AB2`, `AB3`, `AB4`, `AB5` (Adams-Bashforth methods of orders 2-5).
  * `-h`: The integration step size (e.g., `1e-3`, `0.001`).

**Optional Arguments:**

  * `-n <samples>`: Number of random configurations to sample and average (default: 1).
  * `-q`: Use quaternion-based orientation representation instead of rotation matrices.
  * `-w`: Run the model without calculating the ground-truth error (for speed).
  * `-j <step_size>`: Use a different step size for finite-difference Jacobian approximation.
  * `-a <a1> <a2> <a3>`: Specify the three alpha joint angles (must be used with `-b` and `-f`).
  * `-b <b1> <b2> <b3>`: Specify the three beta joint translations (must be used with `-a` and `-f`).
  * `-f <f1> <f2> <f3>`: Specify the three initial force guesses for the solver (must be used with `-a` and `-b`).
  * `--help`: Display the help message with all options.

#### Example Usage

  * **Run a single simulation with Broyden solver, 2nd-order Adams-Bashforth integrator, and a 1mm step size:**

    ```sh
    ./bin -s Broyden -i AB2 -h 1e-3
    ```

  * **Benchmark the performance over 1,000,000 samples using the fully optimized configuration from the report:**

    ```sh
    ./bin -s Broyden -i ForwardEuler -h 1e-3 -q -j 7e-3 -n 1000000 -w
    ```

### Running the Data Collection and Visualization Pipeline

A `Python` pipeline along with a a `YAML` config file is provided to easily run experiments, collect data, and evaluate results. A `data` folder will be automatically created once the pipeline is run; all the outputs will be save to it.

**Setup a virtual environment and install dependencies:**
  It is highly recommended to use a virtual environment so as to not affect your main `Python` runtime.

  ```sh
  python -m venv venv && source venv/bin/activate
  pip install -r requirement.txt
  ```

**Run the pipeline:**
  Please note that with the default values in the config file, the pipeline might take a few hours to a day to finish running:

  ```sh
  python main.py
  ```

## 📂 Code Structure

The project is organized into several key components:

  * `src/`: Contains the `.cpp` source files.
  * `include/`: Contains the `.hpp` header files.
  * `utils/`: Contains the `Python` code used to reproduce the experiments, collect the data, and generate graphs and figures.
  * `legacy/`: Contains the initial `Python` implementation for developing the different methods being used.
  * `report/`: Contains the `Latex` source code and the figures for the final manuscript.
  * `main.py`: A pipeline that is made to facilitate the process of running experiments, collection data, and processing results.
  * `config.yaml`: The config file for the `Python` pipeline.
  * `Makefile`: The build script for compiling the project.
  * `report.pdf`: The manuscript detailing the project's methods and results.

## 🌷 A Note of Gratitude

This project would not have been possible without the incredible guidance and support of my co-authors and mentors. I want to extend my deepest gratitude to **Reinhard M. Grassmann** and **Prof. Jessica Burgner-Kahrs**, whose invaluable mentorship, patience, and expertise shaped this research from start to finish. Thank you for giving me the opportunity to work in the amazing environment of the **Continuum Robotics Laboratory**.

Working with all of you was an absolute privilege and the highlight of my undergraduate career. Thank you for everything.
