import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from experiment import euclidean_pdist, conv1d_full, cross


class TestEuclideanPDist:

    def test_basic(self):
        x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        d = euclidean_pdist(x)
        s = np.sqrt(2)
        assert_allclose(d, np.array([1.0, 1.0, 1.0, s, s, s]), rtol=5e-15)

    def test_calculation(self):
        rng = np.random.default_rng(121263137472525314065)
        m = 100
        n = 8
        x = rng.normal(size=(m, n))
        d = euclidean_pdist(x)
        k = 0
        for i in range(m - 1):
            for j in range(i + 1, m):
                assert_allclose(d[k], np.linalg.norm(x[i] - x[j]), rtol=5e-15)
                k += 1

    def test_with_out(self):
        x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        out = np.empty(6)
        d = euclidean_pdist(x, out=out)
        s = np.sqrt(2)
        assert_allclose(d, np.array([1.0, 1.0, 1.0, s, s, s]), rtol=5e-15)

    def test_bad_out_shape(self):
        x = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        out = np.zeros(7)
        with pytest.raises(ValueError, match=r'does not equal m\*\(m - 1\)/2'):
            euclidean_pdist(x, out=out)

    def test_zero_input_length(self):
        x = np.empty((0, 5))
        with pytest.raises(ValueError, match='must be at least 1'):
            euclidean_pdist(x)

    def test_x_too_big(self):
        x = np.empty((2**40-1, 0))
        with pytest.raises(ValueError, match='too big'):
            euclidean_pdist(x)


class TestConv1dFull:

    def test_basic(self):
        x = np.array([1, 2, 4, 10, -1])
        y = np.array([2, 3, 0, 1])
        z = conv1d_full(x, y)
        assert_equal(z, np.convolve(x, y, mode='full'))

    def test_basic_with_out(self):
        x = np.array([1, 2, 4, 10, -1])
        y = np.array([2, 3, 0, 1])
        out = np.full(len(x) + len(y) - 1, fill_value=np.nan)
        conv1d_full(x, y, out=out)
        assert_equal(out, np.convolve(x, y, mode='full'))

    def test_basic_broadcast(self):
        # x.shape is (3, 6)
        x = np.array([[1, 3, 0, -10, 2, 2],
                      [0, -1, 2, 2, 10, 4],
                      [8, 9, 10, 2, 23, 3]])
        # y.shape is (2, 1, 7)
        y = np.array([[[3, 4, 5, 20, 30, 40, 29]],
                      [[5, 6, 7, 10, 11, 12, -5]]])
        # result should have shape (2, 3, 12)
        result = conv1d_full(x, y)
        assert result.shape == (2, 3, 12)
        for i in range(2):
            for j in range(3):
                assert_equal(result[i, j], np.convolve(x[j], y[i, 0]))

    def test_bad_out_shape(self):
        x = np.ones((1, 2))
        y = np.ones((2, 3))
        out = np.zeros((2, 3))  # Not the correct shape.
        with pytest.raises(ValueError, match=r'does not equal m \+ n - 1'):
            conv1d_full(x, y, out=out)


class TestCross:

    def test_basic_2d(self):
        x = np.array([1.0, 2.5])
        y = np.array([3.0, -0.5])
        z = cross(x, y)
        assert_equal(z[0], np.cross(x, y))

    def test_basic_3d(self):
        x = np.array([2.0, 3.0, -4.0])
        y = np.array([-1.0, 7.0, 5.0])
        z = cross(x, y)
        assert_equal(z, np.cross(x, y))
