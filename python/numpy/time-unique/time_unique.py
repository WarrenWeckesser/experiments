from itertools import combinations_with_replacement
import numpy as np


def work(a):
    return np.unique(a)


def run_single(arrays):
    return [work(a) for a in arrays]


def run_multi(njobs, arrays):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=njobs) as executor:
        job_results = executor.map(work, arrays)

    return job_results


def make_string_arrays(rng, n, size):
    values = ["ABCDEFG", "UVWXYZ", "12345", "99999999999", "plate", "of", "shrimp"]
    return [rng.choice(values, size=size) for _ in range(n)]


def make_vstring_arrays(rng, n, size):
    k = [chr(t) for t in range(945, 970)]
    values = (['', '0123456789', 'plate of shrimp'] + k +
              [''.join(c) for c in combinations_with_replacement(k, 2)] +
              [''.join(c) for c in combinations_with_replacement(k, 3)] +
              [''.join(c) for c in combinations_with_replacement(k, 4)])
    return [rng.choice(values, size=size).astype(np.dtypes.StringDType) for _ in range(n)]


if __name__ == "__main__":
    import timeit

    print(f"numpy {np.__version__}")

    rng = np.random.default_rng(121263137472525314065)

    num_arrays = 12
    n = 5_000_000

    print("Generating random data.")
    # samples = [rng.integers(0, 256, size=n).astype(np.uint8) for _ in range(num_arrays)]
    # samples = [rng.integers(0, 100000, size=n) for _ in range(num_arrays)]
    # samples = make_string_arrays(rng, num_arrays, size=n)
    samples = make_vstring_arrays(rng, num_arrays, size=n)
    print("Done generating data.")
    print(f"data type: {samples[0].dtype}")
    print(f"array size: {n}")
    print()

    for num_threads in range(1, 5):
        print(f"{num_threads = }")

        num_runs = 10
        if num_threads == 1:
            t = timeit.timeit(lambda: run_single(samples),
                            number=num_runs) / num_runs
        else:
            t = timeit.timeit(lambda: run_multi(num_threads, samples),
                            number=num_runs) / num_runs
        print(f"{t = :.4g}")
        print()
