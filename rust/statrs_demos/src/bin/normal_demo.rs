// Example using the `Normal` distribution.

use statrs::distribution::{ContinuousCDF, Normal};

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
    let mu = 0.0;
    let sigma = 1.0;
    let normal = Normal::new(mu, sigma).unwrap();
    let x0 = -35.0;
    let p = normal.cdf(x0);
    print!("normal.cdf(x0)=");
    out(p, true);
    let x1 = normal.inverse_cdf(p);
    print!("normal.inverse_cdf(p)=");
    out(x1, true);
}
