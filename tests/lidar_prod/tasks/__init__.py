import hydra
from omegaconf import OmegaConf


OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
