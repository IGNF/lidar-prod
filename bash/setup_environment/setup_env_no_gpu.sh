set -e

conda install -y mamba -n base -c conda-forge
conda create -y -n validation_module python=3.9 anaconda
conda activate validation_module

conda install -y -c conda-forge python-pdal==3.0.2

pip install --upgrade pip

pip install comet_ml==3.25.0
pip install python-dotenv==0.19.2
pip install hydra-core==1.1.1
pip install hydra-colorlog==1.1.0
pip install geopandas==0.10.2
pip install laspy==2.1.1
pip install rich==11.2.0
pip install pytest==6.2.5
pip install optuna==2.10.0
pip install pyshp==2.2.0
pip install flake8==4.0.1

conda install -y -c conda-forge pdal==2.3.0
mamba install -y pytorch==1.8.1 -c pytorch -c conda-forge
mamba install -y pyg==2.0.3 -c pytorch -c pyg -c conda-forge	# update cudatoolkit to 11.3.1 et pytorch to 1.10.2
mamba install -y pytorch-lightning==1.5.8 -c conda-forge
FORCE_CUDA=0 pip install torch-points-kernels --no-cache	# is angry because no daal==2021.3.0, but pip install flake8==4.0.1 works nonetheless
pip install torchmetrics==0.6.2
