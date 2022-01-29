// Example of the `Weibull` distribution.

use rand::distributions::Distribution;
use rand::thread_rng;
use statrs::distribution::{ContinuousCDF, Weibull};

fn main() {
    let shape = 2.5f64;
    let scale = 1.0f64;
    let weibull = Weibull::new(shape, scale).unwrap();

    println!("RVs:");
    let mut rng = thread_rng();
    for _ in 0..10 {
        println!("{}", weibull.sample(&mut rng));
    }

    println!("CDF:");
    for x in [0.0, 0.1, 0.5, 1.0] {
        println!("{}  {}", x, weibull.cdf(x));
    }
}
