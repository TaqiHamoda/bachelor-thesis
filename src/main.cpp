#include "../include/main.hpp"

using namespace std;


void getOptions(int argc, char **argv, Options &options){
    for(int i = 0; i < argc; i++){
        if(string("--help").compare(argv[i]) == 0){
            cout << "Usage: " << argv[0] << " [-s solver] [-i integrator] [-h step size]\n" <<
            "  -s: Solver (Newton, Broyden)\n" <<
            "  -i: Integrator (ForwardEuler, RK2/3/4, AB2/3/4/5)\n" <<
            "  -h: Step Size (1e-3, 0.001, etc)\n\n" <<
            "  -q [optional]: Use quaternions\n" <<
            "  -j [optional]: Finite Differences Step Size (1e-2, 0.1, etc). Must be larger than h\n" <<
            "  -w [optional]: Run the model without error calculation\n" <<
            "  -n [optional]: Number of samples\n\n" <<
            "  -b [optional]: The three beta values corresponding to inner tube, middle tube, outer tube\n" <<
            "\tIf omitted, a random sample will be chosen. Must accompany alphas\n" <<
            "  -a [optional]: The three beta values corresponding to inner tube, middle tube, outer tube\n" <<
            "\tIf omitted, a random sample will be chosen. Must accompany betas\n" <<
            "  -f [optional]: The three initial forces used by the solver corresponding to inner tube, middle tube, outer tube\n" <<
            "\tIf omitted, a zero vector will be chosen. Must accompany joints\n" <<
            "  -e [optional]: Stiffness matrix values corresponding to EI, GJ of inner tube, middle tube, outer tube\n" <<
            "\tIf omitted, the preset stiffness matrix will be used. Usually used for least squares data fitting\n" <<
            "  -k [optional]: Precurvature values corresponding to inner tube, middle tube, outer tube\n" <<
            "\tIf omitted, the preset precurvature values will be used. Usually used for least squares data fitting\n";
            exit(0);
        }
    }

    char opt;
    u_char flag = 0;
    while ((opt = getopt(argc, argv, "s:i:h:qj:wn:bafek")) != -1) {
        switch (opt) {
            case 's':
                if(string("Newton").compare(optarg) == 0){
                    options.solver = NEWTON_t;
                }
                else if(string("Broyden").compare(optarg) == 0){
                    options.solver = BROYDEN_t;
                }
                else{
                    cerr << "Solver: " << optarg << " not found\n";
                    exit(EXIT_FAILURE);
                }

                flag |= (1 << 2);
                break;
            case 'i':
                if(string("ForwardEuler").compare(optarg) == 0){
                    options.integrator = FORWARDEULER_t;
                }
                else if(string("RK2").compare(optarg) == 0){
                    options.integrator = RK2_t;
                }
                else if(string("RK3").compare(optarg) == 0){
                    options.integrator = RK3_t;
                }
                else if(string("RK4").compare(optarg) == 0){
                    options.integrator = RK4_t;
                }
                else if(string("AB2").compare(optarg) == 0){
                    options.integrator = AB2_t;
                }
                else if(string("AB3").compare(optarg) == 0){
                    options.integrator = AB3_t;
                }
                else if(string("AB4").compare(optarg) == 0){
                    options.integrator = AB4_t;
                }
                else if(string("AB5").compare(optarg) == 0){
                    options.integrator = AB5_t;
                }
                else{
                    cerr << "Integrator: " << optarg << " not found\n";
                    exit(EXIT_FAILURE);
                }

                flag |= (1 << 1);
                break;
            case 'h':
                options.step_size = stod(optarg);

                flag |= (1 << 0);
                break;
            case 'q':
                options.use_quaternions = true;
                break;
            case 'w':
                options.without_error = true;
                break;
            case 'j':
                options.use_fd_step = true;
                options.fd_step = stod(optarg);
                break;
            case 'n':
                options.samples = stoi(optarg);
                break;
            case 'b':
                if(optind + TUBE_NUM - 1 >= argc){
                    cerr << "Missing values for betas\n";
                    exit(EXIT_FAILURE);
                }

                options.use_joints = true;
                for(int i = 0; i < TUBE_NUM; i++){
                    options.betas(i) = stod(argv[optind + i]);
                }

                optind += TUBE_NUM; // Skip beta values
                flag |= (1 << 3);
                break;
            case 'a':
                if(optind + TUBE_NUM - 1 >= argc){
                    cerr << "Missing values for alphas\n";
                    exit(EXIT_FAILURE);
                }

                options.use_joints = true;
                for(int i = 0; i < TUBE_NUM; i++){
                    options.alphas(i) = stod(argv[optind + i]);
                }

                optind += TUBE_NUM; // Skip alpha values

                flag |= (1 << 4);
                break;
            case 'f':
                if(optind + TUBE_NUM - 1 >= argc){
                    cerr << "Missing values for forces\n";
                    exit(EXIT_FAILURE);
                }

                options.use_force = true;
                for(int i = 0; i < TUBE_NUM; i++){
                    options.u_init(i) = stod(argv[optind + i]);
                }

                optind += TUBE_NUM; // Skip force values

                flag |= (1 << 5);
                break;
            case 'e':
                if(optind + 2*TUBE_NUM - 1 >= argc){
                    cerr << "Missing values for stiffness matrix\n";
                    exit(EXIT_FAILURE);
                }

                options.use_kbt = true;
                for(int i = 0; i < TUBE_NUM; i++){
                    options.kbt(2*i) = stod(argv[optind + 2*i]);
                    options.kbt(2*i + 1) = stod(argv[optind + 2*i + 1]);
                }

                optind += 2*TUBE_NUM; // Skip stiffness values
                break;
            case 'k':
                if(optind + TUBE_NUM - 1 >= argc){
                    cerr << "Missing values for precurvatures\n";
                    exit(EXIT_FAILURE);
                }

                options.use_curvature = true;
                for(int i = 0; i < TUBE_NUM; i++){
                    options.kappas(i) = stod(argv[optind + i]);
                }

                optind += TUBE_NUM; // Skip precurvature values
                break;
            default: /* '?' */
                cerr << "Usage: " << argv[0] << " [-s solver] [-i integrator] [-h step size]\n" <<
                "Use --help for more information\n";
                exit(EXIT_FAILURE);
        }
    }

    if((flag & 0b111) != 0b111){
        cerr << "Usage: " << argv[0] << " [-s solver] [-i integrator] [-h step size]\n" <<
        "Use --help for more information\n";
        exit(EXIT_FAILURE);
    }
    else if(options.use_joints && (flag >> 3) != 0b111){
        cerr << "Alpha and Beta values must be provided together. Forces must accompany joints\n" <<
        "Use --help for more information\n";
        exit(EXIT_FAILURE);
    }
}


void applyOptions(CosseratRod &model, Options &options){
    model.setSolver(options.solver);
    model.setIntegrator(options.integrator);
    model.setIntegrationStep(options.step_size);

    model.setQuaternion(options.use_quaternions);

    if(options.use_fd_step){
        model.setFiniteDifferenceStep(options.fd_step);
    }else{
        model.setFiniteDifferenceStep(options.step_size);
    }
}


void runExperiment(Tube ctcr[TUBE_NUM], CosseratRod &model, Options &options){
    if(options.samples == 0){ return; }

    CosseratRod model_gt(ctcr);

    model_gt.setSolver(SOLVER_GT);
    model_gt.setIntegrator(INTEGRATOR_GT);
    model_gt.setIntegrationStep(STEP_SIZE_GT);
    model_gt.setFiniteDifferenceStep(STEP_SIZE_GT);

    Eigen::Matrix<DTYPE_t, 3, 2> joints;
    VectorInput u_init;
    VectorInput u_init_opt;

    DataCollection data;

    DTYPE_t res_err = 0;
    DTYPE_t pos_err = 0;
    DTYPE_t ori_err = 0;

    uint i = 0;
    while (i < options.samples) {
        if(options.use_joints){
            joints.col(0) = options.alphas;
            joints.col(1) = options.betas;
        }else{
            joints = sample_joints(ctcr);
        }

        model.setJoints(joints);
        model_gt.setJoints(joints);

        if(options.use_force){
            u_init = options.u_init;
        }else{
            u_init.setZero();
        }

        u_init_opt = model.solve(u_init);

        if(!options.without_error){
            res_err = model_gt.evaluate(u_init_opt, false).norm();

            Eigen::Matrix<DTYPE_t, 4, 4> tip = model.constructTubeEnd(0);
            Eigen::Matrix<DTYPE_t, 4, 4> tip_gt = model_gt.constructTubeEnd(0);

            pos_err = 1e3*(tip.col(3).head(3) - tip_gt.col(3).head(3)).norm();
            ori_err = ((tip_gt.block<3, 3>(0, 0).transpose()*tip.block<3, 3>(0, 0)).trace() - 1)/2;
        }

        // If collecting multiple data points, ignore invalid configurations.
        if(options.samples > 1 and model.getSolverIterations() == MAX_ITERATION){
            continue;
        }

        data.runtime += model.getRuntime();

        data.residual_error += res_err;
        data.position_error += pos_err;
        data.orientation_error += ori_err;

        data.function_calls += model.getFunctionCalls();
        data.solver_iters += model.getSolverIterations();

        i++;
    }

    cout << "Runtime: " << data.runtime/(DTYPE_t)options.samples << " microseconds\n";

    cout << "Residual Error: " << data.residual_error/(DTYPE_t)options.samples << "\n";
    cout << "Position Error: " << data.position_error/(DTYPE_t)options.samples << " millimetres\n";
    cout << "Orientation Error: " << data.orientation_error/(DTYPE_t)options.samples << " degrees\n";

    cout << "Solver Iterations: " << data.solver_iters/(DTYPE_t)options.samples << "\n";
    cout << "Function Calls: " << data.function_calls/(DTYPE_t)options.samples << "\n";

    if(options.samples == 1){
        cout << "Alphas: " << model.getJoints().col(0).transpose() << "\n";
        cout << "Betas: " << model.getJoints().col(1).transpose() << "\n";
        cout << "U Guess: " << u_init.transpose() << "\n";
        cout << "U Optimal: " << u_init_opt.transpose() << "\n";

        for(i = 0; i < TUBE_NUM; i++){
            cout << "Tip Position of Tube" << i + 1 << " (meters): " << model.constructTubeEnd(i).col(3).head(3).transpose() << "\n";
        }

        cout << "Max Absolute Eigenvalue of A: " << model.getMaxJacobianEigen(u_init_opt) << "\n";
    }

    cout << "\n";
}


int main(int argc, char **argv){
    Options options;
    getOptions(argc, argv, options);

    Tube ctcr[TUBE_NUM];
    setupCTCR(ctcr);


    for(int i = 0; i < TUBE_NUM; i++){
            if(options.use_kbt){
                ctcr[i].kbt.diagonal() << options.kbt(2*i), options.kbt(2*i), options.kbt(2*i + 1);
            }

            if(options.use_curvature){
                ctcr[i].kappa = options.kappas(i);
                ctcr[i].u(0) = options.kappas(i);
            }
    }

    CosseratRod model(ctcr);

    applyOptions(model, options);
    runExperiment(ctcr, model, options);

    return 0;
}
