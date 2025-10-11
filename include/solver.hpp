#pragma once

#include "utils.hpp"


#define TOLERANCE     1e-06  // Based on scipy default solver tolerance
#define MAX_ITERATION 100

#define NEWTON_t  0
#define BROYDEN_t 1

typedef std::function<VectorInput(Evaluate, VectorInput, VectorInput)> _solve;


Eigen::Matrix<DTYPE_t, 3, 3> approxFprime(Evaluate f, VectorInput x, VectorInput f_x);

class Solver
{
    private:
        _solve solver;

        bool initialized = false;
        VectorInput grad;
        Eigen::Matrix<DTYPE_t, 3, 3> jac_inv;

    protected:
        VectorInput Newton(Evaluate f, VectorInput x, VectorInput f_x);
        VectorInput Broyden(Evaluate f, VectorInput x, VectorInput f_x);

    public:
        int idx = 0;
        DTYPE_t error[MAX_ITERATION];

        void setSolver(u_char solver_t);
        VectorInput solve(Evaluate f, VectorInput x);
};
