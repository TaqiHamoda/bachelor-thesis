#include "../include/solver.hpp"

using namespace std::placeholders;


Eigen::Matrix<DTYPE_t, 3, 3> approxFprime(Evaluate f, VectorInput x, VectorInput f_x)
{
    Eigen::Matrix<DTYPE_t, 3, 3> jac;

    VectorInput temp = VectorInput::Zero();
    for(int i = 0; i < 3; i++){
        temp(i) = (DTYPE_t)EPSILON;
        jac.col(i) = (f(x + temp, true) - f_x)/(DTYPE_t)EPSILON;
        temp(i) = 0;
    }

    return jac;
}


void Solver::setSolver(u_char solver_t){
    switch(solver_t){
        case NEWTON_t:
            this->solver = std::bind(&Solver::Newton, this, _1, _2, _3);
            break;
        case BROYDEN_t:
            this->solver = std::bind(&Solver::Broyden, this, _1, _2, _3);
            break;
        default:
            break;
    }
}


// Solver Implementation
VectorInput Solver::solve(Evaluate f, VectorInput x)
{
    this->initialized = false;

    VectorInput f_x = f(x, false);
    if(f_x.norm() < TOLERANCE){
        return x;
    }

    VectorInput dx;
    for(this->idx = 0; this->idx < MAX_ITERATION; this->idx++){
        dx = solver(f, x, f_x);
        x = x - dx;

        error[this->idx] = dx.norm();
        if(error[this->idx] < TOLERANCE){
            this->idx++;
            break;
        }

        f_x = f(x, false);
    }

    return x;
}


// Newton Implementation
VectorInput Solver::Newton(Evaluate f, VectorInput x, VectorInput f_x)
{
    Eigen::Matrix<DTYPE_t, TUBE_NUM, TUBE_NUM> jac = approxFprime(f, x, f_x);
    VectorInput grad = jac.partialPivLu().solve(f_x);

    return grad;
}


// Broyden Implementation
VectorInput Solver::Broyden(Evaluate f, VectorInput x, VectorInput f_x)
{
    // Source: https://misha.fish/archive/docs/484-spring-2019/ch3lec6.pdf

    if(!this->initialized){
        f_x = f(x, true);
        this->jac_inv = approxFprime(f, x, f_x).inverse();
        this->grad = this->jac_inv*f_x;

        this->initialized = true;
        return this->grad;
    }

    VectorInput g = -this->grad;
    VectorInput g_next = this->jac_inv*f_x;

    VectorInput v = g.transpose()*this->jac_inv;
    Eigen::Matrix<DTYPE_t, TUBE_NUM, TUBE_NUM> outer = g_next * v.transpose();
    this->jac_inv -= outer / (g.transpose()*(g + g_next));

    this->grad = this->jac_inv*f_x;

    return this->grad;
}