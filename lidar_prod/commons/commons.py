import logging
from typing import Callable
import warnings
import time

import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf


def extras(config: DictConfig):
    log = logging.getLogger(__name__)
    if config.ignore_warnings:
        log.debug("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")
    if config.print_config:
        print_config(config, resolve=True)


def print_config(
    config: DictConfig, resolve: bool = True, out_f: str = "config_tree.txt"
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        out_f (str, optional): where to save the printed config.

    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in config:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open(out_f, "w") as fp:
        rich.print(tree, file=fp)


def eval_time(function: Callable):
    """decorator to log the duration of the decorated method"""

    def timed(*args, **kwargs):
        log = logging.getLogger(__name__)
        time_start = time.time()
        result = function(*args, **kwargs)
        time_elapsed = round(time.time() - time_start, 2)

        log.info(f"Processing time of {function.__name__}: {time_elapsed}s")
        return result

    return timed
