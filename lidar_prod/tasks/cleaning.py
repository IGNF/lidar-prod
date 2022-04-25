import logging
import os
import os.path as osp
from typing import Iterable, Optional, Union
import pdal

from lidar_prod.tasks.utils import get_pdal_reader, get_pdal_writer

log = logging.getLogger(__name__)


class Cleaner:
    """Keep only necessary extra dimensions channels."""

    def __init__(self, extra_dims: Optional[Union[Iterable[str], str]]):
        """Format extra_dims parameter from config.

        Args:
            extra_dims (Optional[Union[Iterable[str], str]]): each dim should have format dim_name:pdal_type.
            If a string, used directly; if an iterable, dimensions are joined together.

        """

        self.extra_dims = []
        if extra_dims:
            self.extra_dims = extra_dims
            if not isinstance(extra_dims, str):
                self.extra_dims = ",".join(extra_dims)

    def run(self, in_f: str, out_f: str):
        """Clean out LAS extra dimensions.

        Args:
            in_f (str): input LAS path
            out_f (str): output LAS path, with specified extra dims.
        """

        pipeline = pdal.Pipeline()
        pipeline |= get_pdal_reader(in_f)
        pipeline |= get_pdal_writer(out_f, extra_dims=self.extra_dims)
        pipeline.execute()
        os.makedirs(osp.dirname(out_f), exist_ok=True)
        log.info(f"Saved to {out_f}")
