#!/bin/bash

for JOB in dcai_gcb_09_00 dcai_gcb_09_01 dcai_gcb_09_02 dcai_gcb_09_03
do
    sbatch run.sh $JOB
done
