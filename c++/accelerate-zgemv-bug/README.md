The code here demonstrates an issue that is discussed in https://github.com/scipy/scipy/pull/25696.

On MacOS, it was observed in NumPy that computing `CC @ weights` produces
a `nan+nanj` at position 17 of the output, but row 17 of `CC` does not contain
`nan`, and `weights` does not contain nan.

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
