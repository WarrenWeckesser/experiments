// Print all the ways 3 identical items can be distributed in 4 bins.

use combiter::CombIterArray;

fn main() {
    let ci = CombIterArray::<3, 4>::new();
    for c in ci {
        println!("{:?}", c);
    }
}
