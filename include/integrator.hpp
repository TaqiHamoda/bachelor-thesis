#pragma once

#include "utils.hpp"


#define FORWARDEULER_t 0
#define RK2_t          1
#define RK3_t          2
#define RK4_t          3
#define AB2_t          4
#define AB3_t          5
#define AB4_t          6
#define AB5_t          7


typedef std::function<VectorState(Derive, DTYPE_t, VectorState&, DTYPE_t)> _integrate;


class Integrator {
    private:
        DTYPE_t h;

        VectorState f_t[5];
        bool initialized[4] = {false, false, false, false};

        _integrate integrator;

    protected:
        VectorState ForwardEuler(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h);
        VectorState RK2(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h);
        VectorState RK3(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h);
        VectorState RK4(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h);
        VectorState AB2(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h);
        VectorState AB3(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h);
        VectorState AB4(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h);
        VectorState AB5(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h);

    public:
        void setStepSize(DTYPE_t step_size) { h = step_size; };
        DTYPE_t getStepSize() { return h; };
        void setIntegrator(u_char integrator_t);
        VectorState integrate(Derive f, VectorState y0, DTYPE_t t_start, DTYPE_t t_end);
};


const DTYPE_t A0[2] = {3.0/2.0, -1.0/2.0};
const DTYPE_t A1[3] = {23.0/12.0, -16.0/12.0, 5.0/12.0};
const DTYPE_t A2[4] = {55.0/24.0, -59.0/24.0, 37.0/24.0, -9.0/24.0};
const DTYPE_t A3[5] = {1901.0/720.0, -2774.0/720.0, 2616.0/720.0, -1274.0/720.0, 251.0/720.0};
