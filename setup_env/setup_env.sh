#!/bin/bash

# Be sure that you are using last pip version
# by running pip install --upgrade pip

# Run this from a bash using
# source setup_env/setup_env.sh

# Note:

set -e

conda install mamba --yes -n base -c conda-forge  # mamba is a conda on steroids
mamba env create -f setup_env/requirements.yml --force
conda activate lidar_prod

# Troubleshooting:
# Problem: nothing provides __glibc >=2.17,<3.0.a0 needed by libgdal-3.6.2-h10cbb15_3
# If you encounter this error,
# Creating the environment via conda instead of mamba might fix it, depending of your system.
