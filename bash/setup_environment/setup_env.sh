#!/bin/bash

set -e
pip install --upgrade pip

conda install mamba --yes -n base -c conda-forge # mamba is a conda on steroids
mamba env create -f bash/setup_environment/requirements.yml
conda activate lidar_prod