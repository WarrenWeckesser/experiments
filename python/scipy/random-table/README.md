The Python function `random_table(table, size=None, rng=None)` generates a
random contingency table with the same variable sums as the given table.

*Example*

```
In [71]: c = np.array([[36, 5, 14], [14, 10, 21]])

In [72]: c
Out[72]: 
array([[36,  5, 14],
       [14, 10, 21]])
```

Generate one random table:

```
In [73]: random_table(c)
Out[73]: 
array([[28,  6, 21],
       [22,  9, 14]])

```

The `size` parameter allows several random tables to be generated in one call:

```

In [74]: random_table(c, size=3)
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
In [75]: tables = random_table(c, size=100000)

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

Higher dimensional tables are accepted.  Here, `c3` has shape `(2, 2, 3)`:

```
In [37]: c3 = np.array([[[24, 25, 9], [35, 15, 6]], [[24, 9, 8], [25, 8, 12]]])

In [38]: c3
Out[38]: 
array([[[24, 25,  9],
        [35, 15,  6]],

       [[24,  9,  8],
        [25,  8, 12]]])

In [39]: random_table(c3)
Out[39]: 
array([[[23, 22, 11],
        [37, 12,  9]],

       [[23, 10, 10],
        [25, 13,  5]]])
```
