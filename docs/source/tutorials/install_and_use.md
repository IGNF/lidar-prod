# Installation and usage

## Install dependencies

We use [Anaconda(https://anaconda.org/)] to manage and isolate dependencies. 
The provided environment setup script also installs [Mamba](https://mamba.readthedocs.io/en/latest/index.html),
which gets on top of conda for faster environment installs.

```yaml
# clone project
git clone https://github.com/IGNF/lidar-prod-quality-control
cd lidar-prod-quality-control

# install conda
# see https://www.anaconda.com/products/individual

# install postgis to request building database
sudo apt-get install postgis

# create conda environment
source bash/setup_environment/setup_env.sh


# activate the virtual env
conda activate lidar_prod
```

## Use application as a package

To run the module from anywhere, you can install as a package in a your virtual environment.

```bash
# activate your env
conda activate lidar_prod

# install the package
pip install --upgrade https://github.com/IGNF/lidar-prod-quality-control/tarball/prod  # from github directly, using production branch
pip install -e .  # from local sources
```

To run the module as a package, you will need a source cloud point in LAS format with an additional channel containing predicted building probabilities (`ai_building_proba`) and another one containing predictions entropy (`entropy`). The names of thes channel can be specified via hydra config `config.data_format.las_dimensions`.

To run using default configurations of the installed package, use
```bash
python -m lidar_prod.run paths.src_las=</path/to/file.las>
```

You can specify a different yaml config file with the flags `--config-path` and `--config-name`. You can also override specific parameters. By default, results are saved to a `./outputs/` folder, but this can be overriden with `paths.output_dir` parameter. Refer to [hydra documentation](https://hydra.cc/docs/next/tutorials/basic/your_first_app/config_file/) for the overriding syntax.

To print default configuration run `python -m lidar_prod.run -h`. For pretty colors, run `python -m lidar_prod.run print_config=true`.

## Run sequentialy on multiple files

Hydra supports running the python script with several different values for a parameter via a `--multiruns`|`-m` flag and values separated by a comma.

```bash
python -m lidar_prod.run --multiruns paths.src_las=[file_1.las],[file_2.las],[file_3.las]
```