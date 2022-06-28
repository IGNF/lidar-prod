import sys
import os.path as osp
import hydra
from omegaconf import OmegaConf

# Custom resolver to use partial instantiation
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
# Append package's dirname so that app and optimization scripts can import from the package.
sys.path.append(osp.dirname(osp.dirname(__file__)))
