import logging
import time
from typing import List, Sequence

import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf


def print_config(
    config: DictConfig,
    fields: Sequence[str] = ("hydra", "paths", "data_format", "building_validation"),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for (
        field
    ) in fields:  # TODO: try replacing fields by config to print them all in order...
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def eval_time(method):
    """decorator to log the duration of the decorated method"""

    def timed(*args, **kwargs):
        log = logging.getLogger(__name__)
        time_start = time.time()
        result = method(*args, **kwargs)
        time_elapsed = round(time.time() - time_start, 2)

        log.info(f"Processing time of {method.__name__}: {time_elapsed}s")
        return result

    return timed
