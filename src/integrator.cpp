#include "../include/integrator.hpp"

using namespace std::placeholders;


void Integrator::setIntegrator(u_char integrator_t){
    switch(integrator_t){
        case FORWARDEULER_t:
            this->integrator = std::bind(&Integrator::ForwardEuler, this, _1, _2, _3, _4);
            break;
        case RK2_t:
            this->integrator = std::bind(&Integrator::RK2, this, _1, _2, _3, _4);
            break;
        case RK3_t:
            this->integrator = std::bind(&Integrator::RK3, this, _1, _2, _3, _4);
            break;
        case RK4_t:
            this->integrator = std::bind(&Integrator::RK4, this, _1, _2, _3, _4);
            break;
        case AB2_t:
            this->integrator = std::bind(&Integrator::AB2, this, _1, _2, _3, _4);
            break;
        case AB3_t:
            this->integrator = std::bind(&Integrator::AB3, this, _1, _2, _3, _4);
            break;
        case AB4_t:
            this->integrator = std::bind(&Integrator::AB4, this, _1, _2, _3, _4);
            break;
        case AB5_t:
            this->integrator = std::bind(&Integrator::AB5, this, _1, _2, _3, _4);
            break;
        default:
            break;
    }
}

// Integrator Implementation
VectorState Integrator::integrate(Derive f, VectorState y0, DTYPE_t t_start, DTYPE_t t_end)
{
    this->initialized[0] = false;
    this->initialized[1] = false;
    this->initialized[2] = false;
    this->initialized[3] = false;

    VectorState y = y0;
    for(DTYPE_t t = t_start; t < t_end; t += h) {
        y = integrator(f, t, y, h);
    }

    return y;
}


// Forward Euler Implementation
VectorState Integrator::ForwardEuler(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h)
{
    return y + h * f(t, y);
}


// RK2 Implementation
VectorState Integrator::RK2(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h)
{
    // Based on Ralston's method
    const DTYPE_t a21 = 2.0/3.0;
    const DTYPE_t c2 = 2.0/3.0;
    const DTYPE_t b[2] = {1.0/4.0, 3.0/4.0};

    VectorState Y1 = y;

    VectorState f_t_1 = f(t, Y1);
    VectorState Y2 = Y1 + h*a21*f_t_1;

    return Y1 + h*(b[0]*f_t_1 + b[1]*f(t + c2*h, Y2));
}


// RK3 Implementation
VectorState Integrator::RK3(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h)
{
    // Based on Ralston's third-order method
    const DTYPE_t a21 = 1.0/2.0;
    const DTYPE_t a3[2] = {0.0, 3.0/4.0};
    const DTYPE_t b[3] = {2.0/9.0, 1.0/3.0, 4.0/9.0};
    const DTYPE_t c[2] = {1.0/2.0, 3.0/4.0};

    VectorState Y1 = y;

    VectorState f_t_1 = f(t, Y1);
    VectorState Y2 = Y1 + h*a21*f_t_1;

    VectorState f_t_2 = f(t + c[0]*h, Y2);
    VectorState Y3 = Y1 + h*(a3[0]*f_t_1 + a3[1]*f_t_2);

    return Y1 + h*(b[0]*f_t_1 + b[1]*f_t_2 + b[2]*f(t + c[1]*h, Y3));
}


// RK4 Implementation
VectorState Integrator::RK4(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h)
{
    // Based on Ralston's fourth-order method
    const DTYPE_t a21 = 0.4;
    const DTYPE_t a3[2] = {0.29697761, 0.15875964};
    const DTYPE_t a4[3] = {0.21810040, -3.05096516, 3.83286476};
    const DTYPE_t b[4] = {0.17476028, -0.55148066, 1.20553560, 0.17118478};
    const DTYPE_t c[3] = {0.4, 0.45573725, 1.0};

    VectorState Y1 = y;

    VectorState f_t_1 = f(t, Y1);
    VectorState Y2 = Y1 + h*a21*f_t_1;

    VectorState f_t_2 = f(t + c[0]*h, Y2);
    VectorState Y3 = Y1 + h*(a3[0]*f_t_1 + a3[1]*f_t_2);

    VectorState f_t_3 = f(t + c[1]*h, Y3);
    VectorState Y4 = Y1 + h*(a4[0]*f_t_1 + a4[1]*f_t_2 + a4[2]*f_t_3);

    return Y1 + h*(b[0]*f_t_1 + b[1]*f_t_2 + b[2]*f_t_3 + b[3]*f(t + c[2]*h, Y4));
}


// AB2 Implementation
VectorState Integrator::AB2(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h)
{
    if(!this->initialized[0]){
        this->initialized[0] = true;

        this->f_t[0] = f(t, y);
        return y + h*this->f_t[0];
    }

    this->f_t[1] = this->f_t[0];
    this->f_t[0] = f(t, y);

    return y + h*(A0[0]*this->f_t[0] + A0[1]*this->f_t[1]);
}


// AB3 Implementation
VectorState Integrator::AB3(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h)
{
    if(!this->initialized[0]){
        return AB2(f, t, y, h);
    }
    else if(!this->initialized[1]){
        this->initialized[1] = true;

        return AB2(f, t, y, h);
    }

    this->f_t[2] = this->f_t[1];
    this->f_t[1] = this->f_t[0];
    this->f_t[0] = f(t, y);

    return y + h*(A1[0]*this->f_t[0] + A1[1]*this->f_t[1] + A1[2]*this->f_t[2]);
}


// AB4 Implementation
VectorState Integrator::AB4(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h)
{
    if(!this->initialized[0] || !this->initialized[1]){
        return AB3(f, t, y, h);
    }
    else if(!this->initialized[2]){
        this->initialized[2] = true;

        return AB3(f, t, y, h); 
    }

    this->f_t[3] = this->f_t[2];
    this->f_t[2] = this->f_t[1];
    this->f_t[1] = this->f_t[0];
    this->f_t[0] = f(t, y);

    return y + h*(A2[0]*this->f_t[0] + A2[1]*this->f_t[1] + A2[2]*this->f_t[2] + A2[3]*this->f_t[3]);
}


// AB5 Implementation
VectorState Integrator::AB5(Derive f, DTYPE_t t, VectorState &y, DTYPE_t h)
{
    if(!this->initialized[0] || !this->initialized[1] || !this->initialized[2]){
        return AB4(f, t, y, h);
    }
    else if(!this->initialized[3]){
        this->initialized[3] = true;

        return AB4(f, t, y, h);
    }

    this->f_t[4] = this->f_t[3];
    this->f_t[3] = this->f_t[2];
    this->f_t[2] = this->f_t[1];
    this->f_t[1] = this->f_t[0];
    this->f_t[0] = f(t, y);

    return y + h*(A3[0]*this->f_t[0] + A3[1]*this->f_t[1] + A3[2]*this->f_t[2] + A3[3]*this->f_t[3] + A3[4]*this->f_t[4]);
}
