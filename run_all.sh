#!/bin/bash

for JOB in dcai_gcb_08_00 dcai_gcb_08_01 dcai_gcb_08_02 dcai_gcb_08_03 dcai_gcb_08_04 dcai_gcb_08_05 dcai_gcb_08_06 dcai_gcb_08_07
do
    sbatch run.sh $JOB
done
