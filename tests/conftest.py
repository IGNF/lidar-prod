import os
import pytest
from hydra.experimental import compose, initialize


@pytest.fixture
def default_hydra_cfg():
    with initialize(config_path="./../configs/", job_name="config"):
        return compose(config_name="config")
