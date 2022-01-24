combiter
========

The `CombIterArray` in this library provides an iterator over all ways
of distributing `n` identical objects in `m` bins.

Example
-------

Print all the ways of distributing 3 objects in 4 bins:

```
use combiter::CombIterArray;

fn main() {
    let ci = CombIterArray::<3, 4>::new();
    for c in ci {
        println!("{:?}", c);
    }
}
```

Output:
```
[3, 0, 0, 0]
[2, 1, 0, 0]
[2, 0, 1, 0]
[2, 0, 0, 1]
[1, 2, 0, 0]
[1, 1, 1, 0]
[1, 1, 0, 1]
[1, 0, 2, 0]
[1, 0, 1, 1]
[1, 0, 0, 2]
[0, 3, 0, 0]
[0, 2, 1, 0]
[0, 2, 0, 1]
[0, 1, 2, 0]
[0, 1, 1, 1]
[0, 1, 0, 2]
[0, 0, 3, 0]
[0, 0, 2, 1]
[0, 0, 1, 2]
[0, 0, 0, 3]
```