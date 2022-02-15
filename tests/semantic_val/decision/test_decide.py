import pytest

from semantic_val.decision.decide import extract_coor

@pytest.mark.parametrize("las_name, x_span, y_span, buffer, x_min, y_min, x_max, y_max", [
    ("922000_6307000.las", 1000, 1000, 50, 921950, 6305950, 923050, 6307050),
    ("5000_4000_stupid_lasname.las", 1000, 1000, 50, 4950, 2950, 6050, 4050),
])

def test_extract_coor(las_name, x_span, y_span, buffer, x_min, y_min, x_max, y_max):
    assert extract_coor(las_name, x_span, y_span, buffer) == (x_min, y_min, x_max, y_max)