#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate lidar_prod
HYDRA_FULL_ERROR=1

pip install -e .  # install lidar_prod as a package
python -m lidar_prod.run -h  # dry run to test install