// Example using the `Laplace` distribution.

use statrs::distribution::{ContinuousCDF, Laplace};

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
    let a = 0.25;
    let b = 2.5;
    let laplace = Laplace::new(a, b).unwrap();
    let x0 = 0.2;
    let p = laplace.cdf(x0);
    print!("laplace.cdf(x0)=");
    out(p, true);
    let x1 = laplace.inverse_cdf(p);
    print!("laplace.inverse_cdf(p)=");
    out(x1, true);
}
