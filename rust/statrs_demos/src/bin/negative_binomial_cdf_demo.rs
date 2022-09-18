// Example of the cdf function of the `NegativeBinomial` distribution.

use statrs::distribution::{DiscreteCDF, NegativeBinomial};

fn out(x: f64, newline: bool) {
    let a = x.abs();
    if (a == 0.0) || ((a > 1e-6) && (a < 1e6)) {
        print!("{:<20.12}", x);
    } else {
        print!("{:<20.12e}", x);
    }
    if newline {
        println!();
    }
}

fn main() {
    let k = 232u64;
    let r = 26f64;
    let p = 0.0405f64;
    println!("k = {}", k);
    print!("r = ");
    out(r, true);
    print!("p = ");
    out(p, true);
    let negbinom = NegativeBinomial::new(r, p).unwrap();
    let cdf = negbinom.cdf(k);
    print!("cdf = ");
    out(cdf, true);
}
