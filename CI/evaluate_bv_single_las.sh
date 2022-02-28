#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate lidar_prod
HYDRA_FULL_ERROR=1

# NB: debug flag is for single las evaluation.

python -m lidar_prod.run \
print_config=true \
+task="optimize" \
+building_validation.optimization.debug=true \
building_validation.optimization.todo='prepare+evaluate+update' \
building_validation.optimization.paths.input_las_dir=/var/data/cicd/CICD_github_assets/M8.0/20220204_building_val_V0.0_model/20211001_buiding_val_val/ \
building_validation.optimization.paths.results_output_dir=/var/data/cicd/CICD_outputs/opti/ \
building_validation.optimization.paths.building_validation_thresholds_pickle=/var/data/cicd/CICD_github_assets/M8.0/20220204_building_val_V0.0_model/M8.0B2V0.0_buildingvalidation_thresholds.pickle