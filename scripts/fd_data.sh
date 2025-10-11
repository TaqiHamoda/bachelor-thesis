#!/bin/bash

samples=1000000
data_file="./data/fd_step.txt"

run_test(){
    # Rotation Matrix
    args="-s Broyden -i AB4 -h 1e-3 -n $samples -j $1e-3"
    echo "./bin ${args}" >> ${data_file}
    ../bin ${args} >> ${data_file}

    # Quaternion
    args="-s Broyden -i AB4 -h 1e-3 -n $samples -j $1e-3 -q"
    echo "./bin ${args}" >> ${data_file}
    ../bin ${args} >> ${data_file}
}

printf "" > ${data_file}  # Reset data file
for i in $(seq 1 0.25 10)
do
    run_test $i
done
