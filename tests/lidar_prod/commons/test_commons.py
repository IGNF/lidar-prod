import time

from lidar_prod.commons.commons import eval_time


def test_eval_time(caplog):
    durations = [0.1,0.2]
    @eval_time
    def sleeper(d):
        time.sleep(d)
    for d in durations:
        sleeper(d)
        for record in caplog.records:
            print(record)
    