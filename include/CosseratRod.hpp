#pragma once

#include <chrono>
#include <bits/stdc++.h>

#include "utils.hpp"
#include "solver.hpp"
#include "integrator.hpp"


const Eigen::Vector<DTYPE_t, 3> e3(0, 0, 1);

struct StateVariables {
    // Core State Variables
    VectorSpace u0;
    VectorInput uz;
    VectorInput theta;  // The first theta is always zero

    // Auxiliary State Variables
    VectorSpace u_star[TUBE_NUM];
    Eigen::Matrix<DTYPE_t, 3, 3> K;
    Eigen::Matrix<DTYPE_t, 3, 3> ks[TUBE_NUM];
    Eigen::Matrix<DTYPE_t, 3, 3> Rtheta[TUBE_NUM - 1];
};


class CosseratRod {
    private:
        int idx;
        bool quaternion_mode = false;
        bool calculate_J = false;

        DTYPE_t h;
        DTYPE_t fd_step;
        DTYPE_t J_eigen;  // Max Absolute Eigen of Jacobian

        Tube *ctcr;
        DTYPE_t tube_ends[TUBE_NUM];
        DTYPE_t tube_lengths[2*TUBE_NUM + 1];

        VectorState y;
        Eigen::Matrix<DTYPE_t, 2*TUBE_NUM + 1, 2*TUBE_NUM + 1> A;  // Jacobian Matrix
        Eigen::Matrix<DTYPE_t, 2*TUBE_NUM + 1, 2*TUBE_NUM + 1> Bb;

        Eigen::Vector<DTYPE_t, TUBE_NUM> betas;
        Eigen::Vector<DTYPE_t, TUBE_NUM> alphas;

        Solver solver;
        Integrator integrator;

        Derive _derive;
        Evaluate _evaluate;

        VectorState tube_states[TUBE_NUM];

        void init(Tube ctcr[TUBE_NUM]);

        // Equations
        bool isTube(int i);
        bool isStraight(int i);

        Eigen::Vector<DTYPE_t, P_ROT_NUM_R> constructPRotInitR(DTYPE_t psi_0);
        Eigen::Vector<DTYPE_t, P_ROT_NUM_R> constructPRotDotR(VectorState &y0, VectorSpace &u0);

        Eigen::Vector<DTYPE_t, P_ROT_NUM_Q> constructPRotInitQ(DTYPE_t psi_0);
        Eigen::Vector<DTYPE_t, P_ROT_NUM_Q> constructPRotDotQ(VectorState &y0, VectorSpace &u0);

        // Jacobian Stuff
        void constructJacobian(VectorState &y0);
        DTYPE_t calculateMaxJacobianEigen(VectorState y0);

        // Counters and statistics
        int runtime;     // Solver runtime in microseconds
        int func_counter; // How many times the derive function has been called

    public:
        CosseratRod(Tube ctcr[TUBE_NUM]);
        CosseratRod(Tube ctcr[TUBE_NUM], Eigen::Matrix<DTYPE_t, TUBE_NUM, 2> joints, u_char solver, u_char integrator, DTYPE_t h, bool quaternion, DTYPE_t fd_step);

        void setSolver(u_char s){ this->solver.setSolver(s); }
        void setIntegrator(u_char i){ this->integrator.setIntegrator(i); }
        void setIntegrationStep(DTYPE_t h){ this->h = h; }
        void setFiniteDifferenceStep(DTYPE_t j){ this->fd_step = j; }
        void setQuaternion(bool q){ this->quaternion_mode = q; }
        void setJoints(Eigen::Matrix<DTYPE_t, TUBE_NUM, 2> joints);

        StateVariables getStateVariables(VectorState &y0);
        VectorState derive(DTYPE_t t, VectorState &y0);
        VectorInput evaluate(VectorInput x, bool use_fd_step);
        VectorInput solve(VectorInput u_init_guess);
        Eigen::Matrix<DTYPE_t, 4, 4> constructTubeEnd(uint tube_end);

        int getRuntime(){ return this->runtime; }
        int getFunctionCalls(){ return this->func_counter; }
        void resetFunctionCalls(){ this->func_counter = 0; }
        int getSolverIterations(){ return this->solver.idx; }
        DTYPE_t getMaxJacobianEigen(VectorInput x){
            this->calculate_J = true;
            this->evaluate(x, false);
            this->calculate_J = false;

            return this->J_eigen;
        }

        Eigen::Matrix<DTYPE_t, TUBE_NUM, 2> getJoints();
};
