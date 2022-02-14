#!/bin/bash

set -e
conda install mamba -n base -c conda-forge
mamba env create -f bash/setup_environment/requirements.yml
conda activate lidar_prod_module