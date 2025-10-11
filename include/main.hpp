#pragma once

#include <iostream>
#include <chrono>
#include <string>

#include "utils.hpp"
#include "solver.hpp"
#include "integrator.hpp"
#include "CosseratRod.hpp"

#define SOLVER_GT     NEWTON_t  // Ground Truth Solver
#define INTEGRATOR_GT RK4_t     // Ground Truth Integrator
#define STEP_SIZE_GT  1e-5      // Ground Truth Integration Step Size


struct Options {
    // Model Options
    u_char solver;
    u_char integrator;
    DTYPE_t step_size;

    DTYPE_t e = 50e9;

    bool use_quaternions = false;

    bool without_error = false;

    bool use_fd_step = false;
    DTYPE_t fd_step;

    bool use_joints = false;
    VectorInput betas;
    VectorInput alphas;

    bool use_kbt = false;
    Eigen::Vector<DTYPE_t, 2*TUBE_NUM> kbt;

    bool use_curvature = false;
    Eigen::Vector<DTYPE_t, TUBE_NUM> kappas;

    // Experiment Options
    uint samples = 1;

    bool use_force = false;
    VectorInput u_init;

    uint thread_count = 0;
    bool use_data_collection = false;
};


struct DataCollection {
    int runtime = 0;

    DTYPE_t residual_error = 0;
    DTYPE_t position_error = 0;
    DTYPE_t orientation_error = 0;

    DTYPE_t solver_iters = 0;
    DTYPE_t function_calls = 0;
};


void getOptions(int argc, char **argv, Options &options);
void applyOptions(CosseratRod &model, Options &options);
void runExperiment(Tube *ctcr, CosseratRod &model, Options &options);
