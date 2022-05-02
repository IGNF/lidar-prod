# Optimization and evaluation of building validation decision thresholds

Run a multi-objectives hyperparameters optimization of the decision thresholds, to maximize recall and precision directly while also maximizing automation. 

Those thresholds may depend on both the AI model you use and the rule-based classification you are working with.

You need a set of LAS with :
1) A channel with predicted building probability
2) A classification with labels that distinguish false positive, false negative, and true positive from a rules-based building classification. 
 

```bash
conda activate lidar_prod
python lidar_prod/run.py +task=optimize building_validation.optimization.todo='prepare+optimize+evaluate+update' building_validation.optimization.paths.input_las_dir=[path/to/labelled/val/dataset/] building_validation.optimization.paths.results_output_dir=[path/to/save/results] 
```
Nota: to run on a single file during development, add a `+building_validation.optimization.debug=true` flag to the command line.

Optimized decision threshold will be pickled inside the results directory.

To evaluate the optimized module on a test set, change input las folder, and rerun. You need to specify that no optimization is required using the `todo` params. You also need to give the path to the pickled decision trheshold.

```bash
conda activate lidar_prod
python lidar_prod/run.py +task=optimize building_validation.optimization.todo='prepare+evaluate+update' building_validation.optimization.paths.input_las_dir=[path/to/labelled/test/dataset/] building_validation.optimization.paths.results_output_dir=[path/to/save/results] building_validation.optimization.paths.building_validation_thresholds_pickle=[path/to/optimized_thresholds.pickle]
```