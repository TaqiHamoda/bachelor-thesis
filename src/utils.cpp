#include <random>
#include <chrono>

#include "../include/utils.hpp"


DTYPE_t sampleRandom(){
    /*
    Returns a random sample between [0, 1]
    */
    return ((DTYPE_t)rand())/((DTYPE_t)RAND_MAX);
}


Eigen::Matrix<DTYPE_t, TUBE_NUM, 2> sample_joints(Tube ctcr[TUBE_NUM], unsigned int seed)
{
    if(seed != 0){
        srand(seed);  // Seed the RNG
    }
    else{
        auto r = std::chrono::high_resolution_clock::now().time_since_epoch();
        srand(r.count());
    }

    // Initialize the joints matrix.
    // First column is alphas (rotation), second is betas (translation)
    Eigen::Matrix<DTYPE_t, TUBE_NUM, 2> joints;
    for(int i = 0; i < TUBE_NUM; i++){
        joints(i, 0) = M_PI * (2.0 * sampleRandom() - 1);  // Sample alphas from [-pi, pi]
        joints(i, 1) = sampleRandom();  // Sample betas from [0, 1]
    }

    // Construct the transformation matrix to sample betas
    // Based on https://openreview.net/pdf?id=DW9uz_GZ0og
    Eigen::Matrix<DTYPE_t, TUBE_NUM, TUBE_NUM> M_b;
    M_b.setZero();

    DTYPE_t c = -ctcr[TUBE_NUM - 1].l;  // Outer-most tube first
    M_b.col(0).setConstant(c);
    for(int i = 1; i < TUBE_NUM; i++){
        c = ctcr[TUBE_NUM - i].l - ctcr[TUBE_NUM - i - 1].l;
        M_b.col(i).bottomRows(TUBE_NUM - i).setConstant(c);
    }

    joints.col(1) << (M_b*joints.col(1)).reverse();

    return joints;
}


void setupCTCR(Tube ctcr[TUBE_NUM])
{
    // Robot parameters based on the robot from https://openreview.net/pdf?id=DW9uz_GZ0og

    // Data is organized as inner-most tube, middle tube, outer-most tube
    DTYPE_t ls[TUBE_NUM] = {169e-3, 65e-3, 10e-3};  // Length of straight section
    DTYPE_t lc[TUBE_NUM] = {41e-3, 100e-3, 100e-3};  // Length of curve section

    DTYPE_t ro[TUBE_NUM] = {0.5/2*1e-3, 0.9/2*1e-3, 1.5/2*1e-3};  // Outer radius of tube
    DTYPE_t ri[TUBE_NUM] = {0.4/2*1e-3, 0.7/2*1e-3, 1.2/2*1e-3};  // Inner radius of tube

    DTYPE_t kappa[TUBE_NUM] = {28, 12.4, 4.37};  // Curvature of tube

    // Physical parameters of material
    DTYPE_t nu = 0.3;  // Poisson's Ratio
    DTYPE_t E = 50e9;  // Young's Modulus
    DTYPE_t G = E/(2 * (1 + nu));  // Shear Modulus
    DTYPE_t I;  // Moment of Inertia

    for(int i = 0; i < TUBE_NUM; i++){
        ctcr[i].ls = ls[i];
        ctcr[i].lc = lc[i];
        ctcr[i].l = ls[i] + lc[i];

        ctcr[i].ro = ro[i];
        ctcr[i].ri = ri[i];
        ctcr[i].kappa = kappa[i];

        ctcr[i].u << kappa[i], 0, 0;

        I = M_PI * (pow(ctcr[i].ro, 4) - pow(ctcr[i].ri, 4))/4.0;

        // Set the stiffness matrix
        ctcr[i].kbt.setZero();
        ctcr[i].kbt.diagonal() << E*I, E*I, 2*G*I;
    }
}
