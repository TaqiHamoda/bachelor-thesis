#pragma once

#include <math.h>
#include <cmath>
#include <functional>
#include <eigen3/Eigen/Dense>

typedef _Float64 DTYPE_t;

// Definitions
#define EPSILON  1.4901161193847656e-08  // Based on scipy default epsilon

#define TUBE_NUM 3 // Number of tubes

#define P_ROT_NUM_Q 7  // Quaternion Model
#define P_ROT_NUM_R 12 // Rotation Model

#define STATE_NUM_C 7  // The number of constant elements in the state vector
#define STATE_NUM_Q (P_ROT_NUM_Q + STATE_NUM_C)
#define STATE_NUM_R (P_ROT_NUM_R + STATE_NUM_C)
#define STATE_NUM   STATE_NUM_R  // The Maximum state size

typedef Eigen::Vector<DTYPE_t, STATE_NUM_C> VectorStateConstant;
typedef Eigen::Vector<DTYPE_t, STATE_NUM> VectorState;
typedef Eigen::Vector<DTYPE_t, TUBE_NUM> VectorInput;
typedef Eigen::Vector<DTYPE_t, 3> VectorSpace;

typedef std::function<VectorInput(VectorInput, bool)> Evaluate;
typedef std::function<VectorState(DTYPE_t, VectorState &)> Derive;

struct Tube {
    DTYPE_t l;   // Sum of ls and lc
    DTYPE_t ls;  // Length of straight section
    DTYPE_t lc;  // Length of curved section

    DTYPE_t ro;  // Outer radius of tube
    DTYPE_t ri;  // Inner radius of tube

    DTYPE_t kappa;  // Curvature of curved section

    Eigen::Vector<DTYPE_t, 3> u;  // Curvature vector
    Eigen::Matrix<DTYPE_t, 3, 3> kbt;  // Stiffness Matrix
};


void setupCTCR(Tube ctcr[TUBE_NUM]);
Eigen::Matrix<DTYPE_t, TUBE_NUM, 2> sample_joints(Tube ctcr[TUBE_NUM], unsigned int seed = 0);
