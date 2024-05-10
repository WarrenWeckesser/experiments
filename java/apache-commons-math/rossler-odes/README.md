This directory contains an example of using the Apache Commons Math library
to solve the Rossler system of differential equations.

After building the example, a plot can be generated with, for example, Gnuplot,
as follows (in Linux):

```
$ java RosslerSolver > rossler.dat
$ gnuplot -e 'set term svg; set output "rossler.svg"; plot "rossler.dat" using 3:2 with lines'
```

The Gnuplot command generates this plot:

![Rossler plot](https://github.com/WarrenWeckesser/experiments/blob/main/java/apache-commons-math/rossler-odes/rossler.svg)
