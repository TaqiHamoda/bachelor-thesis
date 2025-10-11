#include "../include/CosseratRod.hpp"

using namespace std::placeholders;


CosseratRod::CosseratRod(Tube ctcr[TUBE_NUM]){
    this->init(ctcr);
}


CosseratRod::CosseratRod(Tube ctcr[TUBE_NUM], Eigen::Matrix<DTYPE_t, TUBE_NUM, 2> joints, u_char solver, u_char integrator, DTYPE_t h, bool quaternion, DTYPE_t fd_step){
    this->init(ctcr);

    this->quaternion_mode = quaternion;

    this->solver.setSolver(solver);
    this->integrator.setIntegrator(integrator);

    this->h = h;
    this->fd_step = fd_step;

    this->setJoints(joints);

    this->Bb.setIdentity();
    this->Bb.diagonal().head(TUBE_NUM - 1).setZero();
}


void CosseratRod::init(Tube ctcr[TUBE_NUM]){
    this->ctcr = ctcr;

    this->_derive = std::bind(&CosseratRod::derive, this, _1, _2);
    this->_evaluate = std::bind(&CosseratRod::evaluate, this, _1, _2);
}


Eigen::Matrix<DTYPE_t, TUBE_NUM, 2> CosseratRod::getJoints(){
    Eigen::Matrix<DTYPE_t, 3, 2> joints;
    joints.col(0) = this->alphas;
    joints.col(1) = this->betas;

    return joints;
}


void CosseratRod::setJoints(Eigen::Matrix<DTYPE_t, TUBE_NUM, 2> joints){
    this->alphas = joints.col(0);
    this->betas = joints.col(1);

    this->tube_lengths[0] = 0;
    for(int i = 1; i < TUBE_NUM + 1; i++){
        this->tube_ends[i - 1] = ctcr[i - 1].l + this->betas(i - 1);

        this->tube_lengths[i] = ctcr[i - 1].ls + this->betas(i - 1);
        this->tube_lengths[TUBE_NUM + i] = ctcr[i - 1].l + this->betas(i - 1);
    }

    std::sort(this->tube_lengths, this->tube_lengths + 2*TUBE_NUM + 1);

    for(int i = 0; i < 2*TUBE_NUM + 1; i++){
        if(this->tube_lengths[i] == 0){
            this->idx = i;
            break;
        }
    }
}


Eigen::Vector<DTYPE_t, P_ROT_NUM_Q> CosseratRod::constructPRotInitQ(DTYPE_t psi_0){
    // Quaternion Model
    Eigen::Vector<DTYPE_t, 4> q_init;
    q_init << (DTYPE_t)cos(psi_0/2.0),
              0,
              0,
              (DTYPE_t)sin(psi_0/2.0);

    Eigen::Vector<DTYPE_t, P_ROT_NUM_Q> p_rot;
    p_rot << Eigen::Vector<DTYPE_t, 3>::Zero(),
             q_init;

    return p_rot;
}


Eigen::Vector<DTYPE_t, P_ROT_NUM_Q> CosseratRod::constructPRotDotQ(VectorState &y0, VectorSpace &u0){
    // Quaternion Model
    Eigen::Matrix<DTYPE_t, 4, 3> Wt;
    Wt << -y0(4), -y0(5), -y0(6),
           y0(3), -y0(6),  y0(5),
           y0(6),  y0(3), -y0(4),
          -y0(5),  y0(4),  y0(3);

    Eigen::Vector<DTYPE_t, P_ROT_NUM_Q> p_rot_dot;
    p_rot_dot << 2*(y0(4)*y0(6) + y0(3)*y0(5)),
             2*(y0(5)*y0(6) - y0(3)*y0(4)),
             y0(3)*y0(3) - y0(4)*y0(4) - y0(5)*y0(5) + y0(6)*y0(6),
             Wt*u0/2;  // q_dot = W.T*u0/2

    return p_rot_dot;
}


Eigen::Vector<DTYPE_t, P_ROT_NUM_R> CosseratRod::constructPRotInitR(DTYPE_t psi_0){
    DTYPE_t c_psi = cos(psi_0);
    DTYPE_t s_psi = sin(psi_0);

    Eigen::Matrix<DTYPE_t, 3, 3> R_init;
    R_init << c_psi, -s_psi, 0,
              s_psi,  c_psi, 0,
                  0,      0, 1;

    Eigen::Vector<DTYPE_t, P_ROT_NUM_R> p_rot;
    p_rot << Eigen::Vector<DTYPE_t, 3>::Zero(),
             R_init.reshaped(9, 1);

    return p_rot;
}


Eigen::Vector<DTYPE_t, P_ROT_NUM_R> CosseratRod::constructPRotDotR(VectorState &y0, VectorSpace &u0){
    Eigen::Matrix<DTYPE_t, 3, 3> R = y0.segment(3, 9).reshaped(3, 3);

    Eigen::Matrix<DTYPE_t, 3, 3> R_dot;
    R_dot <<     0, -u0(2),  u0(1),
             u0(2),      0, -u0(0),
            -u0(1),  u0(0),      0;  // u_hat
    R_dot = R*R_dot;

    Eigen::Vector<DTYPE_t, P_ROT_NUM_R> p_rot_dot;
    p_rot_dot << R.col(2),
             R_dot.reshaped(9, 1);

    return p_rot_dot;
}


Eigen::Matrix<DTYPE_t, 4, 4> CosseratRod::constructTubeEnd(uint tube_end){
    Eigen::Matrix<DTYPE_t, 4, 4> tip;
    tip.setZero();

    VectorState s = this->tube_states[tube_end];

    if(this->quaternion_mode){ // Quaternion Model
        tip(0, 0) = s(3)*s(3) + s(4)*s(4) - s(5)*s(5) - s(6)*s(6);
        tip(1, 0) = 2 * (s(4)*s(5) + s(3)*s(6));
        tip(2, 0) = 2 * (s(4)*s(6) - s(3)*s(5));

        tip(0, 1) = 2 * (s(4)*s(5) - s(3)*s(6));
        tip(1, 1) = s(3)*s(3) - s(4)*s(4) + s(5)*s(5) - s(6)*s(6);
        tip(2, 1) = 2 * (s(5)*s(6) - s(3)*s(4));

        tip(0, 2) = 2 * (s(4)*s(6) + s(3)*s(5));
        tip(1, 2) = 2 * (s(5)*s(6) + s(3)*s(4));
        tip(2, 2) = s(3)*s(3) - s(4)*s(4) - s(5)*s(5) + s(6)*s(6);

        // Construct Position
        tip.col(3) << s.head(3),
                    1;
    }
    else{  // Rotation Model
        tip.block<3, 3>(0, 0) = s.segment(3, 9).reshaped(3, 3);
        tip.col(3) << s.head(3),
                    1;
    }

    return tip;
}


bool CosseratRod::isTube(int i){
    return (this->tube_lengths[this->idx] + this->tube_lengths[this->idx + 1])/2 < this->betas(i) + this->ctcr[i].l;
}


bool CosseratRod::isStraight(int i){
    return (this->tube_lengths[this->idx] + this->tube_lengths[this->idx + 1])/2 < this->betas(i) + this->ctcr[i].ls;
}


StateVariables CosseratRod::getStateVariables(VectorState &y0){
    StateVariables state;

    // Prepare info to be used
    VectorInput psi;

    if(this->quaternion_mode){
        state.uz = y0.segment(P_ROT_NUM_Q, 3);
        psi = y0.segment(P_ROT_NUM_Q + 3, 3);
    }else{
        state.uz = y0.segment(P_ROT_NUM_R, 3);
        psi = y0.segment(P_ROT_NUM_R + 3, 3);
    }

    // Prepare Tube Info
    if(!this->isStraight(0)){
        state.u_star[0] = this->ctcr[0].u;
    }
    else{
        state.u_star[0].setZero();
    }

    state.ks[0] = this->ctcr[0].kbt;
    state.u0 = state.ks[0]*state.u_star[0];

    state.theta[0] = 0;
    state.K = state.ks[0];

    DTYPE_t c_theta, s_theta;
    for(int i = 1; i < TUBE_NUM; i++){
        state.theta[i] = psi(i) - psi(0);

        c_theta = cos(state.theta[i]);
        s_theta = sin(state.theta[i]);

        state.Rtheta[i - 1] << c_theta, -s_theta, 0,
                         s_theta,  c_theta, 0,
                         0,        0,       1;

        if(this->isTube(i) && !this->isStraight(i)){
            state.u_star[i] = this->ctcr[i].u;
        }
        else{
            state.u_star[i].setZero();
        }
        
        if(this->isTube(i)){
            state.ks[i] = this->ctcr[i].kbt;
        }
        else{
            state.ks[i].setZero();
        }

        state.K += state.ks[i];
        state.u0 += state.Rtheta[i - 1]*state.ks[i]*state.u_star[i] - state.ks[i](2, 2)*(state.uz[i] - state.uz[0])*e3;
    }

    state.u0 = state.u0.cwiseQuotient(state.K.diagonal());

    return state;
}


void CosseratRod::constructJacobian(VectorState &y0){
    this->A.setZero();
    this->A.col(TUBE_NUM + 1).head(TUBE_NUM - 1).setConstant(-1);
    this->A.topRightCorner(TUBE_NUM - 1, TUBE_NUM - 1).diagonal().setConstant(1);

    StateVariables state = this->getStateVariables(y0);

    DTYPE_t c_theta, s_theta;
    for(int i = 0; i < TUBE_NUM; i++){
        c_theta = cos(state.theta[i]);
        s_theta = sin(state.theta[i]);

        if(i > 0){
            // d u_dot_1,x / d theta_i
            this->A(TUBE_NUM - 1, i - 1) = state.uz[i]*state.u0[0]*c_theta + state.uz[i]*state.u0[1]*s_theta;
            this->A(TUBE_NUM - 1, i - 1) *= state.ks[i](0, 0);

            this->A(TUBE_NUM - 1, i - 1) += state.ks[i](2, 2)*(state.uz[i] - state.u_star[i][2])*(-state.u0[0]*c_theta - state.u0[1]*s_theta);

            // d u_dot_1,y / d theta_i
            this->A(TUBE_NUM, i - 1) = -state.uz[i]*state.u0[0]*s_theta + state.uz[i]*state.u0[1]*c_theta;
            this->A(TUBE_NUM, i - 1) *= state.ks[i](0, 0);

            this->A(TUBE_NUM, i - 1) -= state.ks[i](2, 2)*(state.uz[i] - state.u_star[i][2])*(-state.u0[0]*s_theta + state.u0[1]*c_theta);
        }

        // d u_dot_1,x / d u_1,x and d u_dot_1,x / d u_1,y
        this->A(TUBE_NUM - 1, TUBE_NUM - 1) += state.ks[i](0, 0)*(state.uz[i]*s_theta) - state.ks[i](2, 2)*s_theta*(state.uz[i] - state.u_star[i][2]);
        this->A(TUBE_NUM - 1, TUBE_NUM) += state.ks[i](0, 0)*(state.uz[i] - state.uz[0] - state.uz[i]*c_theta) + state.ks[i](2, 2)*c_theta*(state.uz[i] - state.u_star[i][2]);

        // d u_dot_1,x / d u_i,z
        this->A(TUBE_NUM - 1, TUBE_NUM + 1 + i) = state.u0[1] + state.u0[0]*s_theta - state.u0[1]*c_theta + state.u_star[i][1];
        this->A(TUBE_NUM - 1, TUBE_NUM + 1 + i) *= state.ks[i](0, 0);

        this->A(TUBE_NUM - 1, TUBE_NUM + 1 + i) -= state.ks[i](2, 2)*(state.u0[0]*s_theta - state.u0[1]*c_theta);

        // d u_dot_1,y / d u_i,z
        this->A(TUBE_NUM, TUBE_NUM + 1 + i) = -state.u0[0] + state.u0[0]*c_theta + state.u0[1]*s_theta - state.u_star[i][0];
        this->A(TUBE_NUM, TUBE_NUM + 1 + i) *= state.ks[i](0, 0);

        this->A(TUBE_NUM, TUBE_NUM + 1 + i) -= state.ks[i](2, 2)*(state.u0[0]*c_theta + state.u0[1]*s_theta);

        // d u_dot_i,z / d theta_i
        if(i > 0){
            this->A(TUBE_NUM + 1 + i, i - 1) = state.u_star[i][1]*(-state.u0[0]*s_theta + state.u0[1]*c_theta);
            this->A(TUBE_NUM + 1 + i, i - 1) += state.u_star[i][0]*(state.u0[0]*c_theta + state.u0[1]*s_theta);
        }

        // d u_dot_i,z / d u_1,x and d u_dot_i,z / d u_1,y
        this->A(TUBE_NUM + 1 + i, TUBE_NUM - 1) = state.u_star[i][1]*c_theta + state.u_star[i][0]*s_theta;
        this->A(TUBE_NUM + 1 + i, TUBE_NUM) = state.u_star[i][1]*s_theta - state.u_star[i][0]*c_theta;

        // Multiply by EI/GJ
        this->A.row(TUBE_NUM + 1 + i) *= state.ks[i](0, 0)/state.ks[i](2, 2);
    }

    this->A.row(TUBE_NUM - 1) /= -state.K(0, 0);
    this->A.row(TUBE_NUM) /= -state.K(0, 0);

    this->A(TUBE_NUM, TUBE_NUM) = this->A(TUBE_NUM - 1, TUBE_NUM - 1);
    this->A(TUBE_NUM, TUBE_NUM - 1) = -this->A(TUBE_NUM - 1, TUBE_NUM);
}


DTYPE_t CosseratRod::calculateMaxJacobianEigen(VectorState y0){
    this->constructJacobian(y0);
    return this->A.eigenvalues().cwiseAbs().maxCoeff();
}


VectorState CosseratRod::derive(DTYPE_t t, VectorState &y0){
    if(this->calculate_J){
        this->J_eigen = std::max(this->J_eigen, this->calculateMaxJacobianEigen(y0));
    }

    this->func_counter++;

    VectorInput uz_dot;
    VectorSpace u[TUBE_NUM - 1];
    StateVariables state = this->getStateVariables(y0);

    uz_dot(0) = state.ks[0](0, 0)/state.ks[0](2, 2)*(state.u0(0)*state.u_star[0](1) - state.u0(1)*state.u_star[0](0));
    for(int i = 1; i < TUBE_NUM; i++){
        u[i] = state.Rtheta[i - 1].transpose()*state.u0 + (state.uz(i) - state.uz(0))*e3;

        if(this->isTube(i)){
            uz_dot(i) = state.ks[i](0, 0)/state.ks[i](2, 2)*(u[i](0)*state.u_star[i](1) - u[i](1)*state.u_star[i](0));
        }
        else{
            uz_dot(i) = 0;
        }
    }

    VectorState y_dot;
    if(this->quaternion_mode){
        y_dot << this->constructPRotDotQ(y0, state.u0),
              uz_dot,
              state.uz,
              1,
              Eigen::Vector<DTYPE_t, STATE_NUM_R - STATE_NUM_Q>::Zero();
    }else{
        y_dot << this->constructPRotDotR(y0, state.u0),
              uz_dot,
              state.uz,
              1;
    }

    return y_dot;
}


VectorInput CosseratRod::evaluate(VectorInput x, bool use_fd_step){
    // If the solver jacobian is being built with a different step size
    if (use_fd_step) {
        this->integrator.setStepSize(this->fd_step);
    }
    else{
        this->integrator.setStepSize(this->h);
    }

    VectorInput psi_init = this->alphas + x.cwiseProduct(this->betas.cwiseAbs());

    if(this->quaternion_mode){
        this->y << constructPRotInitQ(psi_init(0)),
         x,
         psi_init,
         0,
         Eigen::Vector<DTYPE_t, STATE_NUM_R - STATE_NUM_Q>::Zero();
    }
    else{
        this->y << constructPRotInitR(psi_init(0)),
         x,
         psi_init,
         0;
    }

    this->J_eigen = -1;  // Reset maximum eigenvalue
    uint tube_end = TUBE_NUM - 1;  // Index to check for which tube end the integration is at

    int start_idx = this->idx;
    for(; this->idx < 2*TUBE_NUM; this->idx++){
        this->y = this->integrator.integrate(this->_derive, this->y, this->tube_lengths[this->idx], this->tube_lengths[this->idx + 1]);

        if(this->tube_lengths[this->idx + 1] == this->tube_ends[tube_end]){
            this->tube_states[tube_end] = this->y;
            tube_end--;
        }
    }
    this->idx = start_idx;

    if(this->calculate_J){
        this->J_eigen = std::max(this->J_eigen, this->calculateMaxJacobianEigen(this->y));
    }

    VectorInput u_forces;

    if(this->quaternion_mode){
        u_forces = this->y.segment(P_ROT_NUM_Q, TUBE_NUM);
    }
    else{
        u_forces = this->y.segment(P_ROT_NUM_R, TUBE_NUM);
    }

    return u_forces;
}


VectorInput CosseratRod::solve(VectorInput u_init_guess){
    this->func_counter = 0;
    this->calculate_J = false;

    auto start = std::chrono::high_resolution_clock::now();
    VectorInput res = this->solver.solve(this->_evaluate, u_init_guess);
    auto end = std::chrono::high_resolution_clock::now();

    auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    this->runtime = runtime.count();

    return res;
}
