// Example of the `NegativeBinomial` distribution.

use statrs::distribution::{Discrete, DiscreteCDF, NegativeBinomial};

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
    let r = 13.1f64;
    let p = 0.7f64;
    print!("r =");
    out(r, false);
    print!("   p = ");
    out(p, true);
    let negbinom = NegativeBinomial::new(r, p).unwrap();
    println!("k pmf                 cdf");
    for k in 0..5 {
        let pmf = negbinom.pmf(k);
        let cdf = negbinom.cdf(k);
        print!("{} ", k);
        out(pmf, false);
        out(cdf, true);
    }

    let mut sum = 0.0f64;
    let nterms = 200;
    for k in 0..nterms {
        sum += negbinom.pmf(nterms - k - 1);
    }
    print!("Sum of first {} PMF values: ", nterms);
    out(sum, true);
}
