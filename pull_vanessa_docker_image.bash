#!/bin/bash
module load singularity/2.4.5
export SINGULARITY_CACHEDIR=$SCRATCH/.singularity
cd $SCRATCH
singularity pull docker://vanessa/sherlock:recurrent_nn

