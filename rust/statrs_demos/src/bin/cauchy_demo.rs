// Example of the Cauchy distribution.

use statrs::distribution::{ContinuousCDF, Cauchy};
use statrs::statistics::Distribution;

fn main() {
    let location = -5.0f64;
    let scale = 100.0f64;
    let cauchy = Cauchy::new(location, scale).unwrap();

    println!("Cauchy(location={}, scale={})", location, scale);
    println!("Entropy:");
    println!("{}", cauchy.entropy().unwrap());
    println!("CDF:");
    for x in [-1e10, -5e6, -1.1e-4] {
        println!("{}  {}", x, cauchy.cdf(x));
    }
}
