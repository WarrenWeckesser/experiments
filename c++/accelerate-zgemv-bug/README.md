The code here demonstrates an issue that is discussed in https://github.com/scipy/scipy/pull/25696.  There are cases where the product of a matrix containing some occurrences of `inf+nanj` is generating `nan` in the output at positions where the corresponding row of the matrix contains only finite values.

*Update* A simple C program, `demo_issue.c`, demonstrates the unexpected behavior.
It is a self-contained example that uses a matrix with shape (32, 8). It does not
require the NumPy arrays in the `.npy` files here.

Build and run `demo_issue`:

```
% clang demo_issue.c -DACCELERATE_NEW_LAPACK -framework Accelerate -o demo_issue
% ./demo_issue 
num_output_nans = 31  (expected 30)

These two output values were not expected to contain nan:
y[0] = (0.000000,0.000000)
y[31] = (nan,nan)
```

-----
On MacOS, it was observed in NumPy that computing `CC @ weights` produces
a `nan+nanj` at position 17 of the output, but row 17 of `CC` does not contain
`nan`, and `weights` does not contain nan.  The value `inf+nanj` occurs in `CC`,
but not in row 17.

The code in `check.cpp` uses `cblas_zgemv` to compute `(CC @ weights)[17]` and
`CC[17,:] @ weights`.  They should give the same value, but the first gives `nan+nanj`
and the second gives a finite complex value:

```
% ./check
CC has shape (75, 37)
@ is matrix multiplication computed with cblas_zgemv.

(CC @ weights)[17] = (nan,nan)
CC[17,:] @ weights = (0.693874,0.368663)
```
