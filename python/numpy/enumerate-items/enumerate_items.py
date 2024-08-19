import numpy as np


def enumerate_items(x, start=0):
    # x is expected to be a 1D sequence (e.g. numpy ndarray, pandas Series).
    items, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    max_count = counts.max()
    seq = np.arange(start, start + max_count)
    c = np.empty(x.size, dtype=int)
    for index, count in enumerate(counts):
        c[inv == index] = seq[:count]
    return c


if __name__ == "__main__":
    x = np.array([4, 1, 2, 2, 2, 3, 4, 2, 1, 4, 4, 1])
    c = enumerate_items(x)
    print(x)
    print(c)
