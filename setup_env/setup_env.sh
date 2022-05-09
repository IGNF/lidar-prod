#!/bin/bash

# Be sure that you are using last pip version
# by running pip install --upgrade pip

# Run this from a bash using
# source setup_env/setup_env.sh


set -e

conda install mamba --yes -n base -c conda-forge # mamba is a conda on steroids
mamba env create -f setup_env/requirements.yml --force
conda activate lidar_prod