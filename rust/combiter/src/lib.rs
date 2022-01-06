/// The `CombIter` iterator iterates through all combinations
/// of `TOTAL` objects distributed among `NBINS` bins.
/// The iterated item has type `[usize; NBINS]`.

// TODO: Ensure that the edge case NBINS=0 is handled properly
// (which first requires deciding what that means).

pub struct CombIter<const TOTAL: usize, const NBINS: usize> {
    counts: [usize; NBINS],
}

impl<const TOTAL: usize, const NBINS: usize> CombIter<TOTAL, NBINS> {
    pub fn new() -> Self {
        Self { counts: [0; NBINS] }
    }
}

impl<const TOTAL: usize, const NBINS: usize> Default for CombIter<TOTAL, NBINS> {
    fn default() -> Self {
        Self { counts: [0; NBINS] }
    }
}

impl<const TOTAL: usize, const NBINS: usize> Iterator for CombIter<TOTAL, NBINS> {
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
        let ci = CombIter::<2, 3>::new();
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
        let ci = CombIter::<2, 3>::default();
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
        let ci = CombIter::<2, 4>::new();
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
        let ci = CombIter::<3, 3>::new();
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
        let ci = CombIter::<3, 4>::new();
        for (c, expectedc) in ci.zip(expected.iter()) {
            assert_eq!(&c, expectedc);
        }
    }

    #[test]
    fn test_basic_0_3() {
        let expected = [[0, 0, 0]];
        let ci = CombIter::<0, 3>::new();
        for (c, expectedc) in ci.zip(expected.iter()) {
            assert_eq!(&c, expectedc);
        }
    }

    #[test]
    fn test_basic_5_1() {
        let expected = [[5]];
        let ci = CombIter::<5, 1>::new();
        for (c, expectedc) in ci.zip(expected.iter()) {
            assert_eq!(&c, expectedc);
        }
    }

    #[test]
    fn test_last_5_4() {
        let ci = CombIter::<5, 4>::new();
        let last = ci.last().unwrap();
        assert_eq!(last, [0, 0, 0, 5]);
    }
}
