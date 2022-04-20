import logging
import time

from lidar_prod.commons.commons import eval_time

# See https://docs.pytest.org/en/latest/how-to/logging.html#caplog-fixture
def test_eval_time(caplog):
    caplog.set_level(logging.INFO)
    durations = [0.1,0.2]

    @eval_time
    def sleeper(d):
        time.sleep(d)

    for d in durations:
        sleeper(d)
        assert caplog.records[-1].message.endswith(f"{d}s")
    