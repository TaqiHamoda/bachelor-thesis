#!/bin/bash

samples=1000000
data_file="./data/runtime.txt"

solvers=(Newton Broyden)
integrators=(ForwardEuler AB2 AB3 AB4 AB5 RK2 RK3 RK4)

run_test(){
    # Rotation Matrix
    args="-s $1 -i $2 -h 1e-3 -n $samples"
    echo "./bin ${args}" >> ${data_file}
    ../bin ${args} >> ${data_file}

    # Quaternion
    args="-s $1 -i $2 -h 1e-3 -n $samples -q"
    echo "./bin ${args}" >> ${data_file}
    ../bin ${args} >> ${data_file}
}


printf "" > ${data_file}  # Reset data file
for solver in "${solvers[@]}"
do
    for integrator in "${integrators[@]}"
    do
        run_test $solver $integrator
    done
done
