import pytest
import numpy as np
from scipy import ndimage
from scipy.ndimage import maximum_position


def ident(x):
    return x


def index0(x):
    return x[0]


@pytest.mark.parametrize('func', [ndimage.maximum_position,
                                  ndimage.minimum_position])
@pytest.mark.parametrize('index, get', [(1, ident),
                                        ([1], index0)])
def test_case_bug_report(func, index, get):
    # From gh-25279
    img = np.zeros((10, 10))
    img[2:5, 2:5] = 1.0   # plateau under label 1
    img[7:9, 7:9] = 5.0   # values under label 2 — should be irrelevant for index=[1]

    labels = np.zeros((10, 10), dtype=int)
    labels[2:5, 2:5] = 1
    labels[7:9, 7:9] = 2

    pos_with    = get(func(img, labels, index))

    img_zeroed = img.copy()
    img_zeroed[labels == 2] = 0   # change only pixels outside label 1
    pos_without = get(func(img_zeroed, labels, index))

    # print("img:")
    # print(img)
    # print("labels:")
    # print(labels)
    assert pos_with == (2, 2)
    assert pos_without == (2, 2)



@pytest.mark.parametrize('func, ref', [(ndimage.maximum_position, (1, 4)),
                                       (ndimage.minimum_position, (0, 0))])
def test_maximum_position_index_scalar_vs_list(func, ref):
    x = np.array([[2, 4, 2, 0, 1],
                  [2, 1, 0, 1, 4],
                  [3, 2, 3, 2, 4]])

    lbls = np.array([[ 5, -5,  5, 15, 15],
                     [-5, 15, 15, -5,  5],
                     [15, 15, 15, -5,  5]])

    p1 = func(x, lbls, index=5)
    p2 = func(x, lbls, index=[5])

    assert p1 == ref
    assert p2[0] == ref


@pytest.mark.parametrize('func', [ndimage.maximum_position,
                                  ndimage.minimum_position])
@pytest.mark.parametrize('index, get', [(1, ident),
                                        ([1], index0)])
def test_maximum_position_outside_label_invariance(func, index, get):
    # gh-25279: the reported maximum position of a label must depend only on
    # that label's own pixels. An unstable sort used to let values belonging
    # to other labels reorder the tie-break among equal maxima, so the result
    # changed when unrelated pixels changed.

    # Use xp = np for now...
    xp = np
    dtype = np.float32

    labels = xp.asarray([[0, 0, 0, 0],
                         [0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 0, 0, 2]])

    # label 1 is a plateau of equal maxima; only the label-2 pixel differs.
    input_a = xp.asarray([[0, 0, 0, 0],
                          [0, 1, 1, 0],
                          [0, 1, 1, 0],
                          [0, 0, 0, 0]], dtype=dtype)
    input_b = xp.asarray([[0, 0, 0, 0],
                          [0, 1, 1, 0],
                          [0, 1, 1, 0],
                          [0, 0, 0, 9]], dtype=dtype)
    pos_a = get(func(input_a, labels, xp.asarray(index)))
    pos_b = get(func(input_b, labels, xp.asarray(index)))
    assert pos_a == (1, 1)
    assert pos_b == (1, 1)


def test_maximum_position_outside_label_invariance2():
    # Use xp = np for now...
    xp = np
    dtype = np.float32

    labels = xp.asarray([[2, 2, 2, 2, 2],
                         [2, 1, 1, 1, 0],
                         [2, 1, 1, 1, 0],
                         [2, 2, 2, 2, 2]])
    # label 1 is a plateau of equal maxima; only the label-2 pixel differs.
    input_a = xp.asarray([[0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0]], dtype=dtype)
    input_b = xp.asarray([[8, 9, 8, 9, 8],
                          [9, 1, 1, 1, 8],
                          [9, 1, 1, 1, 8],
                          [9, 8, 9, 8, 9]], dtype=dtype)
    pos_a = ndimage.maximum_position(input_a, labels, xp.asarray([1]))
    pos_b = ndimage.maximum_position(input_b, labels, xp.asarray([1]))
    assert pos_a == (1, 1)
    assert pos_b == (1, 1)


@pytest.mark.parametrize('func', [ndimage.maximum_position,
                                  ndimage.minimum_position])
@pytest.mark.parametrize('index, get', [(1, ident),
                                        ([1], index0)])
def test_maximum_position_outside_label_invariance3(func, index, get):
    # Use xp = np for now...
    xp = np
    dtype = np.float32

    labels = xp.asarray([[2, 2, 2, 1, 1],
                         [2, 0, 1, 1, 0],
                         [2, 1, 1, 0, 0],
                         [1, 1, 2, 2, 2]])
    # label 1 is a plateau of equal maxima; only the label-2 pixel differs.
    input_a = xp.asarray([[0, 0, 0, 1, 1],
                          [0, 0, 1, 1, 0],
                          [0, 1, 1, 0, 0],
                          [1, 1, 0, 0, 0]], dtype=dtype)
    input_b = xp.asarray([[9, 9, 9, 1, 1],
                          [0, 0, 1, 1, 0],
                          [0, 1, 1, 0, 0],
                          [1, 1, 9, 9, 9]], dtype=dtype)
    pos_a = get(func(input_a, labels, xp.asarray(index)))
    pos_b = get(func(input_b, labels, xp.asarray(index)))
    assert pos_a == (0, 3)
    assert pos_b == (0, 3)
