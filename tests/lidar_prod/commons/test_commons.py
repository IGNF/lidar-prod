import logging
import os
import tempfile
import time
from hydra.experimental import compose, initialize

from lidar_prod.commons.commons import eval_time, print_config

# See https://docs.pytest.org/en/latest/how-to/logging.html#caplog-fixture
def test_eval_time(caplog):
    caplog.set_level(logging.INFO)
    durations = [0.1, 0.2]

    @eval_time
    def sleeper(d):
        time.sleep(d)

    for d in durations:
        sleeper(d)
        assert caplog.records[-1].message.endswith(f"{d}s")


def test_file_existence():
    with tempfile.TemporaryDirectory() as td:
        out_f = os.path.join(td, "config_tree.txt")
        with initialize(
            config_path=os.path.join("./../../../", "configs/"), job_name="config"
        ):
            cfg = compose(config_name="config")
            print_config(
                cfg,
                resolve=False,
                out_f=out_f,
            )
            assert os.path.exists(out_f)
