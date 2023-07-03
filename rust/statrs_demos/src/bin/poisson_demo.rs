use rand;
use rand::distributions::Distribution;
use statrs::distribution::{Discrete, Poisson};

fn main() {
    let lam = 1e6;
    let pois = Poisson::new(lam).unwrap();
    let target = (lam + 1e3) as u64;

    let mut r = rand::thread_rng();
    let nsamples = 10000000;
    let mut target_count = 0;
    for _ in 0..nsamples {
        let sample = pois.sample(&mut r);
        if sample == target as f64 {
            target_count += 1
        }
    }
    let p = pois.pmf(target);
    println!("lam = {}", lam);
    println!("target = {}", target);
    println!("p({}) = {}", target, p);
    println!("nsamples = {}", nsamples);
    println!("Expected count: {}", p * (nsamples as f64));
    println!("Actual count:   {}", target_count);
}
