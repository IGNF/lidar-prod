import logging
import os
import os.path as osp
import pdal

log = logging.getLogger(__name__)


class Cleaner:
    """Discard unnecessary extra dimensions channels."""

    def __init__(self, keep_extra_dims):
        self.keep_extra_dims = ",".join(keep_extra_dims)

    def run(self, in_f: str, out_f: str):
        self.setup(out_f)
        self.update(in_f, out_f)

    def setup(self, out_f):
        """Setup step : create output dir.

        Args:
            out_f (_type_): outputh file path.
        """
        os.makedirs(osp.dirname(out_f), exist_ok=True)

    def update(self, in_f: str, out_f: str):
        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader(in_f, type="readers.las")
        pipeline |= pdal.Writer.las(
            filename=out_f, forward="all", extra_dims=self.keep_extra_dims
        )
        pipeline.execute()
        log.info(f"Saved to {out_f}")
