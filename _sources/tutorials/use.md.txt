# Using the app

## Run within a docker container

Up to date docker images (named `lidar_prod`) are created via Github integration actions (see [Developer's guide](../guides/development.md)). 

To run the app, use

```bash
docker run \
-v {local_src_las_dir}:/inputs/ \
-v {local_output_dir}:/outputs/
lidar_prod \
python lidar_prod/run.py \
paths.src_las=/inputs/{src_las_basename}.las
paths.output_dir=/outputs/
# + other options...

```

A docker image encapsulating the virtual environment and application sources can also be built using the provided Dockerfile. This Dockerfile is not standalone and should be part of the repository (whose content is copied into the image), on the github reference you want to build from.

## Run as a python module
To run the module as a module, you will need a source cloud point in LAS format with an additional channel containing predicted building probabilities (`ai_building_proba`) and another one containing predictions entropy (`entropy`). The names of thes channel can be specified via hydra config `config.data_format.las_dimensions`.

To run using default configurations of the installed module, use
```bash
python -m lidar_prod.run paths.src_las=</path/to/file.las>
```

You can specify a different yaml config file with the flags `--config-path` and `--config-name`. You can also override specific parameters. Overriding `building_validation.application.shp_path` will force the use of the provided shapefile instead of querying DB Uni to build a shapefile on the fly. By default, results are saved to a `./outputs/` folder, but this can be overriden with `paths.output_dir` parameter. Refer to [hydra documentation](https://hydra.cc/docs/next/tutorials/basic/your_first_app/config_file/) for the overriding syntax.

To print default configuration run `python -m lidar_prod.run -h`. For pretty colors, run `python -m lidar_prod.run print_config=true`.

## Run from source directly

For developments and debugging, you can run the package directly from python sources instead:

```bash
# activate an env matching ./bash/setup_env.sh requirements.
conda activate lidar_prod
python lidar_prod/run.py paths.src_las=[/path/to/file.las]
```
## Run Different tasks

Different tasks can be runned from the application. Currently, they are:
- `apply_on_building` to identify building
- `identify_vegetation_unclassified` to identify unclassified and vegetation
- `optimize_building` to evaluate the best parameters for building detection
- `optimize_veg_id` to evaluate the best parameters for vegetation detection
- `optimize_unc_id` to evaluate the best parameters for unclassified detection
- `get_shapefile` to create a shapefile from the BD UNI corresponding to a las file
- `cleaning` to prepare las file with the correct dimension (mostly useful in development)

To use on of those tasks, simply add "+task=[task_name]" to the options list, like this:
```bash
python lidar_prod/run.py +task=apply_on_building ...
```
