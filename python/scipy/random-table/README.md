The Python module `random_table.py` defines the functions `random_table()`
and `random_table_from_table()`.  Both generate random contingency tables
using the same underlying method.  The only difference is in the inputs
to the functions.

The Python function `random_table_from_table(table, size=None, rng=None)`
generates a random contingency table with the same variable sums as the
given table.

The function `random_table(*sums, size=None, rng=None)` accepts the
quantities associated with the levels of each categorical variable
(i.e. the marginal sums for each variable).

*Examples*

```
In [71]: c = np.array([[36, 5, 14], [14, 10, 21]])

In [72]: c
Out[72]: 
array([[36,  5, 14],
       [14, 10, 21]])
```

Generate one random table:

```
In [73]: random_table_from_table(c)
Out[73]: 
array([[28,  6, 21],
       [22,  9, 14]])

```

The `size` parameter allows several random tables to be generated in one call:

```

In [74]: random_table_from_table(c, size=3)
Out[74]: 
array([[[28,  7, 20],
        [22,  8, 15]],

       [[27, 10, 18],
        [23,  5, 17]],

       [[26,  7, 22],
        [24,  8, 13]]])
```

Generate a large number of tables, and compare the sample mean to the expected
mean computed by `scipy.stats.contingency.expected_freq`:

```
In [75]: tables = random_table_from_table(c, size=100000)

In [76]: tables.mean(axis=0)
Out[76]: 
array([[27.49068,  8.24867, 19.26065],
       [22.50932,  6.75133, 15.73935]])

In [77]: from scipy.stats.contingency import expected_freq

In [78]: expected_freq(c)
Out[78]: 
array([[27.5 ,  8.25, 19.25],
       [22.5 ,  6.75, 15.75]])
```

Higher dimensional tables are accepted.  Here, `c3` has shape `(2, 3, 3)`.
We use `scipy.stats.contingency.margins` to compute the marginal sums
for `c3`.

```
In [85]: c3 = np.array([[[24, 15, 9], [26, 15, 6], [5, 11, 14]],
    ...:                [[40, 11, 7], [21, 10, 12], [9, 8, 7]]])

In [86]: c3
Out[86]:
array([[[24, 15,  9],
        [26, 15,  6],
        [ 5, 11, 14]],

       [[40, 11,  7],
        [21, 10, 12],
        [ 9,  8,  7]]])

In [87]: from scipy.stats.contingency import margins

In [88]: margins(c3)
Out[88]:
[array([[[125]],

        [[125]]]),
 array([[[106],
         [ 90],
         [ 54]]]),
 array([[[125,  70,  55]]])]
```
Generate a random table from `c3`.
```
In [89]: sample = random_table_from_table(c3)

In [90]: sample
Out[90]:
array([[[20, 15, 10],
        [21, 14, 12],
        [22,  6,  5]],

       [[28, 16, 17],
        [21, 14,  8],
        [13,  5,  3]]])
```
Verify that the marginal sums of `sample` are the same as those of `c3`:
```
In [91]: margins(sample)
Out[91]:
[array([[[125]],

        [[125]]]),
 array([[[106],
         [ 90],
         [ 54]]]),
 array([[[125,  70,  55]]])]
```

Instead of an existing contingency table, the function `random_table()`
accepts the marginal sums associated with each categorical variable.
For example,

```
In [94]: random_table([10, 10, 20], [15, 25])
Out[94]:
array([[ 3,  7],
       [ 4,  6],
       [ 8, 12]])
```
Note that the sums of the rows are [10, 10, 20], and the sums of
the columns are [15, 25].

Higher-dimensional tables can also be generated with this function.
The following generates one contingency table with shape (3, 2, 3):
```
In [95]: table = random_table([10, 10, 20], [15, 25], [6, 28, 6])

In [96]: table
Out[96]:
array([[[0, 3, 0],
        [0, 6, 1]],

       [[1, 3, 0],
        [0, 5, 1]],

       [[2, 4, 2],
        [3, 7, 2]]])
```
Verify that the margins have the required sums.

```
In [97]: [t.ravel() for t in margins(table)]
Out[97]: [array([10, 10, 20]), array([15, 25]), array([ 6, 28,  6])]
```
