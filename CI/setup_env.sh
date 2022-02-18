#!/bin/bash

# Set up a lidar_prod virtual env for the following tests
eval "$(conda shell.bash hook)"
conda deactivate
source ./bash/setup_environment/setup_env.sh