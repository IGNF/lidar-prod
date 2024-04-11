# Installation

## Set up a virtual environment

We use [Anaconda](https://anaconda.org/)] to manage and isolate dependencies.
The provided environment setup script also installs [Mamba](https://mamba.readthedocs.io/en/latest/index.html),
which gets on top of conda for faster environment installs.

```yaml
# clone project
git clone https://github.com/IGNF/lidar-prod-quality-control
cd lidar-prod-quality-control

# install conda
# see https://www.anaconda.com/products/individual

# you need to install postgis to request a public database
sudo apt-get install postgis

# create conda environment
source setup_env/setup_env.sh

# activate the virtual env
conda activate lidar_prod
```

## Install the app as a python module

To run the application from anywhere, you can install as a module in a your virtual environment.

```bash
# activate your env
conda activate lidar_prod

# install the package from github directly, using production branch
pip install --upgrade https://github.com/IGNF/lidar-prod-quality-control/tarball/prod

```

During development, install in editable mode directly from source with
 ```bash
 pip install --editable .
 ```

Then, refert to the [usage page](./use.md).

## Provide credentials
To help identify buildings, the BD_UNI database is used. To provide credentials, copy bd_uni_connection_params/credentials_template.yaml to bd_uni_connection_params/credentials.yaml
```cp bd_uni_connection_params/credentials_template.yaml bd_uni_connection_params/credentials.yaml```
Then fill the blanks in the file, specifically "user" and "pwd".
