import logging
import os
import tempfile
import time
import warnings

import pytest

from lidar_prod.commons.commons import eval_time, extras, ignore_warnings, print_config


@pytest.mark.xfail
def test_extras(default_hydra_cfg):
    # Will fail since hydra default config cannot be resolved
    # This is done to ensure coverage
    default_hydra_cfg.ignore_warnings = False
    default_hydra_cfg.print_config = False
    extras(default_hydra_cfg)
    default_hydra_cfg.ignore_warnings = True
    default_hydra_cfg.print_config = True
    extras(default_hydra_cfg)


# See https://docs.python.org/3/library/warnings.html#testing-warnings
def test_ignore_warnings():
    def fxn():
        warnings.warn("A fake warning!", Warning)

    with warnings.catch_warnings(record=True) as w:
        # Trigger a warning.
        fxn()
        assert len(w) == 1
        assert issubclass(w[-1].category, Warning)
        # Ignore warnings
        ignore_warnings()
        # Trigger another warning.
        fxn()
        # Still one ?
        assert len(w) == 1


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


def test_file_existence(default_hydra_cfg):
    with tempfile.TemporaryDirectory() as td:
        out_f = os.path.join(td, "config_tree.txt")
        print_config(
            default_hydra_cfg,
            resolve=False,
            out_f=out_f,
        )
        assert os.path.exists(out_f)
