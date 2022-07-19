#!/bin/bash

# Necessary for Bash shells
. /etc/profile

# Tensorflow optimized for A100 with CUDA 11
module load conda/2022-07-01

# Activate conda env
conda activate /lus/theta-fs0/projects/AIASMAAR/dhgpu
