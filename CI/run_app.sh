#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate lidar_prod
HYDRA_FULL_ERROR=1

python -m lidar_prod.run print_config=true \
paths.src_las=/var/data/cicd/CICD_github_assets/M8.0/20220204_building_val_V0.0_model/subsets/871000_6617000_subset_with_probas.las \
paths.output_dir=/var/data/cicd/CICD_outputs/app/ \
data_format.codes.candidates.building='[19, 20, 110, 112, 114, 115]' \
building_validation.application.building_validation_thresholds_pickle=/var/data/cicd/CICD_github_assets/M8.0/20220204_building_val_V0.0_model/M8.0B2V0.0_buildingvalidation_thresholds.pickle