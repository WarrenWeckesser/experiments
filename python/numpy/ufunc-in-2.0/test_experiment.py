import pytest
import numpy as np
from numpy.testing import assert_equal
from experiment import deadzone


@pytest.mark.parametrize('dt', [np.float32, np.float64])
def test_deadzone_types(dt):
    x = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]).astype(dt)
    low = np.array([-1.0], dtype=dt)
    high = np.array([1.0], dtype=dt)
    y = deadzone(x, low, high)
    expected = np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0],
                        dtype=dt)
    assert_equal(y, expected, strict=True)
