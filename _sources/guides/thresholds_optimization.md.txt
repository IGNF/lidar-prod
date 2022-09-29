# How to optimize building validation decision thresholds?

This guide explains how to optimize decision thresholds following the strategy in [this note](../background/thresholds_optimization_process.md).

## Requirements

To optimize the decision thresholds you must be able to evaluate the level of automation that can be reached on data that matches production data. As a result, you need to have _corrected_ data i.e. data of which a rule-based classification was corrected and for which you keep track of the corrections that were made. For building validation, the classification must have codes to distinguish false positive, false negative, and true positive. Theses codes may be configured with parameter `buildings_correction_labels` under configuration group `bulding_validation.optimization`.

Furthermore, the point cloud data must include predictions from the deep learning model trained to detect buildings. This consists in two channels : a `building` channel with predicted probabilities and an `entropy` channel.

A large validation dataset might help having a better sense of the app performances. We used 15km² of corrected data to optimize thresholds, but a larger set might provide more diversity. This being said, performance on an unseen test set was almost equal to performance on the validation set, which indicates a robust evaluation for such volume of data. 


## Running thresholds optimization

### Finding optimal thresholds

> Refer to the [installation tutorial](../tutorials/install.md) to set up your python environment.

Your corrected data must live in a single `input_las_dir` directory as a set of LAS files. 
Prepared and updated files will be saved in subfolder of a `results_output_dir` directory (`./prepared` and `./updated/`, respectively).
They will keep the same basename as the original files.
Be sure that the `data_format` configurations match your data, and in particular the (clasification) `codes` and `las_dimensions` configuration groups.
A `todo` string parameter specifies the steps to run by including 1 or more of the following keywords: `prepare` | `otpimize` | `evaluate` | `update`. 

Run the full optimization module with

```bash
conda activate lidar_prod

python lidar_prod/run.py \
++task=optimize_building \
building_validation.optimization.todo='prepare+optimize+evaluate+update' \
building_validation.optimization.paths.input_las_dir=[path/to/labelled/val/dataset/] \
building_validation.optimization.paths.results_output_dir=[path/to/save/results] 
```

### Evaluation of optimized thresholds on a test set

Once an optimal solution was found, you may want to evaluate the decision process on unseen data to evaluate generalization capability. For that, you will need another test folder of corrected data in the same format as before (a different `input_las_dir`). You need to specify that no optimization is required using the `todo` params. You also need to give the path to the pickled decision thresholds from the previous step, and specify a different `results_output_dir` so that prepared data of test and val test are not pooled together.


```bash
conda activate lidar_prod

python lidar_prod/run.py \
++task=optimize_building \
building_validation.optimization.todo='prepare+evaluate+update' \
building_validation.optimization.paths.input_las_dir=[path/to/labelled/test/dataset/] \
building_validation.optimization.paths.results_output_dir=[path/to/save/results] \
building_validation.optimization.paths.building_validation_thresholds_pickle=[path/to/optimized_thresholds.pickle]
```

### Utils

Debug mode: to run on a single file during development, add a `+building_validation.optimization.debug=true` flag to the command line.


Reference:
- [Deb et al. (2002) - A fast and elitist multiobjective genetic algorithm\: NSGA-II](https://ieeexplore.ieee.org/document/996017)).
