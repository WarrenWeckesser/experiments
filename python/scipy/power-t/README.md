power-t
-------

The function `power_t.find_n()` does the same computation as `power.t.test()` in R for the
case where `n` is computed.

For example:

```
In [12]: find_n(alpha=0.01, power=0.95, sigma=16, delta=1, alternative='two-sided')
Out[12]: 9122.51106485781
```

In R:

```
> power.t.test(delta=1, sd=16, sig.level=0.01, power=0.95, alternative="two.sided")

     Two-sample t test power calculation

              n = 9122.511
          delta = 1
             sd = 16
      sig.level = 0.01
          power = 0.95
    alternative = two.sided

NOTE: n is number in *each* group
```

Using `statsmodels`, with `effect_size = delta / sigma`:

```
In [5]: from statsmodels.stats.power import TTestIndPower

In [6]: TTestIndPower().solve_power(effect_size=1/16, alpha=0.01, power=0.95, alternative='two-sided')
Out[6]: 9122.511064857801
```
