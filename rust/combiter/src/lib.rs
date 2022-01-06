/// The `CombIterArray` iterator iterates through all combinations
/// of `TOTAL` objects distributed among `NBINS` bins.
/// The iterated item has type `[usize; NBINS]`.

// `CombIterArray` passes around arrays on the stack.  It will
// likely crash if `NBINS` is big enough.  Each return value from
// the `next()` method of the `Iterator` trait is a new array, not
// a reference to the array stored in `CombIterArray`.

// TODO: Ensure that the edge case NBINS=0 is handled properly
// (which first requires deciding what that means).

pub struct CombIterArray<const TOTAL: usize, const NBINS: usize> {
    counts: [usize; NBINS],
}

impl<const TOTAL: usize, const NBINS: usize> CombIterArray<TOTAL, NBINS> {
    pub fn new() -> Self {
        Self { counts: [0; NBINS] }
    }
}

impl<const TOTAL: usize, const NBINS: usize> Default for CombIterArray<TOTAL, NBINS> {
    fn default() -> Self {
        Self { counts: [0; NBINS] }
    }
}

impl<const TOTAL: usize, const NBINS: usize> Iterator for CombIterArray<TOTAL, NBINS> {
    type Item = [usize; NBINS];

    fn next(&mut self) -> Option<Self::Item> {
        if self.counts[NBINS - 1] == TOTAL {
            return None;
        }
        // Find the rightmost nonzero bin.
        let mut rightmost_nonzero = NBINS - 1;
        let mut found_nonzero = false;
        for i in (0..NBINS).rev() {
            if self.counts[i] != 0 {
                rightmost_nonzero = i;
                found_nonzero = true;
                break;
            }
        }
        if !found_nonzero {
            // All the values in self.counts are 0.  This means
            // this is the first call of next().  The first iterate
            // is [TOTAL, 0, 0, ,,,, 0].
            self.counts[0] = TOTAL;
            return Some(self.counts);
        }
        if rightmost_nonzero != NBINS - 1 {
            self.counts[rightmost_nonzero] -= 1;
            self.counts[rightmost_nonzero + 1] += 1;
        } else {
            let mut j = rightmost_nonzero - 1;
            while self.counts[j] == 0 {
                j -= 1;
            }
            let p = self.counts[NBINS - 1];
            self.counts[NBINS - 1] = 0;
            self.counts[j + 1] = p + 1;
            self.counts[j] -= 1;
        }
        Some(self.counts)
    }

    // XXX Maybe this could be simpler: accept `self` instead of `mut self`,
    // ignore `self`, and return an appropriately initialized array?
    //
    fn last(mut self) -> Option<Self::Item> {
        for i in 0..NBINS - 1 {
            self.counts[i] = 0;
        }
        self.counts[NBINS - 1] = TOTAL;
        Some(self.counts)
    }
}

//
// Alternate version
//
// `CombData` holds a `Vec`, so the actual data is on the heap.
//
// `CombData` does not implement `Iterator`.  Instead, it
// has a `next()` method that must be called explicitly.
// `next()` returns slices of the `Vec` held by the `CombData`.
//
// (My rust-fu is not strong enough to figure out how to use an
// `Iterator` here.  In an `Iterator`, the values returned by
// `next()` must outlive the `Iterator` object, so, for example,
// they can't be slices of the `counts` vector.  Some other object
// must hold the vector (or array), and it must outlive the `Iterator`.
// A different approach that shows up while looking into this is
// the "streaming iterator".)

pub struct CombData<const TOTAL: usize, const NBINS: usize> {
    counts: Vec<usize>,
}

impl<const TOTAL: usize, const NBINS: usize> CombData<TOTAL, NBINS> {
    pub fn new() -> Self {
        Self {
            counts: vec![0usize; NBINS],
        }
    }

    fn next<'a>(&'a mut self) -> Option<&'a [usize]> {
        if self.counts[NBINS - 1] == TOTAL {
            return None;
        }
        // Find the rightmost nonzero bin.
        let mut rightmost_nonzero = NBINS - 1;
        let mut found_nonzero = false;
        for i in (0..NBINS).rev() {
            if self.counts[i] != 0 {
                rightmost_nonzero = i;
                found_nonzero = true;
                break;
            }
        }
        if !found_nonzero {
            // All the values in self.counts are 0.  This means
            // this is the first call of next().  The first iterate
            // is [TOTAL, 0, ..., 0].
            self.counts[0] = TOTAL;
            return Some(&self.counts);
        }
        if rightmost_nonzero != NBINS - 1 {
            self.counts[rightmost_nonzero] -= 1;
            self.counts[rightmost_nonzero + 1] += 1;
        } else {
            let mut j = rightmost_nonzero - 1;
            while self.counts[j] == 0 {
                j -= 1;
            }
            let p = self.counts[NBINS - 1];
            self.counts[NBINS - 1] = 0;
            self.counts[j + 1] = p + 1;
            self.counts[j] -= 1;
        }
        Some(&self.counts)
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Tests
// - - - - - - - - - - - - - - - - - - - - - - - - - - -

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_2_3() {
        let expected = [
            [2, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 2, 0],
            [0, 1, 1],
            [0, 0, 2],
        ];
        let ci = CombIterArray::<2, 3>::new();
        for (c, expectedc) in ci.zip(expected.iter()) {
            assert_eq!(&c, expectedc);
        }
    }

    #[test]
    fn test_default_2_3() {
        let expected = [
            [2, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 2, 0],
            [0, 1, 1],
            [0, 0, 2],
        ];
        let ci = CombIterArray::<2, 3>::default();
        for (c, expectedc) in ci.zip(expected.iter()) {
            assert_eq!(&c, expectedc);
        }
    }

    #[test]
    fn test_basic_2_4() {
        let expected = [
            [2, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 2, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 2, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 2],
        ];
        let ci = CombIterArray::<2, 4>::new();
        for (c, expectedc) in ci.zip(expected.iter()) {
            assert_eq!(&c, expectedc);
        }
    }

    #[test]
    fn test_basic_3_3() {
        let expected = [
            [3, 0, 0],
            [2, 1, 0],
            [2, 0, 1],
            [1, 2, 0],
            [1, 1, 1],
            [1, 0, 2],
            [0, 3, 0],
            [0, 2, 1],
            [0, 1, 2],
            [0, 0, 3],
        ];
        let ci = CombIterArray::<3, 3>::new();
        for (c, expectedc) in ci.zip(expected.iter()) {
            assert_eq!(&c, expectedc);
        }
    }

    #[test]
    fn test_basic_3_4() {
        let expected = [
            [3, 0, 0, 0],
            [2, 1, 0, 0],
            [2, 0, 1, 0],
            [2, 0, 0, 1],
            [1, 2, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 2, 0],
            [1, 0, 1, 1],
            [1, 0, 0, 2],
            [0, 3, 0, 0],
            [0, 2, 1, 0],
            [0, 2, 0, 1],
            [0, 1, 2, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 2],
            [0, 0, 3, 0],
            [0, 0, 2, 1],
            [0, 0, 1, 2],
            [0, 0, 0, 3],
        ];
        let ci = CombIterArray::<3, 4>::new();
        for (c, expectedc) in ci.zip(expected.iter()) {
            assert_eq!(&c, expectedc);
        }
    }

    #[test]
    fn test_basic_0_3() {
        let expected = [[0, 0, 0]];
        let ci = CombIterArray::<0, 3>::new();
        for (c, expectedc) in ci.zip(expected.iter()) {
            assert_eq!(&c, expectedc);
        }
    }

    #[test]
    fn test_basic_5_1() {
        let expected = [[5]];
        let ci = CombIterArray::<5, 1>::new();
        for (c, expectedc) in ci.zip(expected.iter()) {
            assert_eq!(&c, expectedc);
        }
    }

    #[test]
    fn test_last_5_4() {
        let ci = CombIterArray::<5, 4>::new();
        let last = ci.last().unwrap();
        assert_eq!(last, [0, 0, 0, 5]);
    }

    #[test]
    fn test_combdata_2_3() {
        let expected = [
            [2, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 2, 0],
            [0, 1, 1],
            [0, 0, 2],
        ];
        let mut ci = CombData::<2, 3>::new();
        let mut i = 0;
        while let Some(c) = ci.next() {
            assert_eq!(c, expected[i]);
            i += 1;
        }
    }
}
