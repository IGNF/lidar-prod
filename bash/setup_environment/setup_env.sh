#!/bin/bash

# Be sure that you are using last pip version
# by running pip install --upgrade pip

set -e

conda install mamba --yes -n base -c conda-forge # mamba is a conda on steroids
mamba env create -f bash/setup_environment/requirements.yml --force
conda activate lidar_prod