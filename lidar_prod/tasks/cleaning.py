import logging
import os
import os.path as osp
import pdal

log = logging.getLogger(__name__)


class Cleaner:
    """Keep only necessary extra dimensions channels."""

    def __init__(self, extra_dims):
        self.extra_dims = ",".join(extra_dims)

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
            filename=out_f,
            forward="all",
            extra_dims=self.extra_dims,
            minor_version=4,
            dataformat_id=8,
        )
        pipeline.execute()
        log.info(f"Saved to {out_f}")
