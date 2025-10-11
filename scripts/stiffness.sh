#!/bin/bash

threads=8
files_per_thread=20
data_points_per_file=125000
file_prefix="./data/stiffness"

args="-s Broyden -i AB4 -h 1e-3"

run_test(){
    for f in $(seq 1 1 $files_per_thread)
    do
        data_file="${file_prefix}$1-$f.txt"
        printf "" > ${data_file}  # Reset data file

        for j in $(seq 1 1 $data_points_per_file)
        do
            echo "./bin ${args}" >> ${data_file}
            ../bin ${args} $1 >> ${data_file}
            echo "Data File $1 ($f/$files_per_thread): $j/$data_points_per_file"
        done
    done
}

merge_datafiles(){
    for i in $(seq 1 1 $threads)
    do
        target_file="${file_prefix}$i.txt"

        printf "" > $target_file
        for f in $(seq 1 1 $files_per_thread)
        do
            data_file="${file_prefix}$i-$f.txt"

            cat $data_file >> $target_file
            # rm $data_file  # Usually commented out to have backups
        done
    done
}


for i in $(seq 1 1 $threads)
do
    run_test $i &
done

merge_datafiles