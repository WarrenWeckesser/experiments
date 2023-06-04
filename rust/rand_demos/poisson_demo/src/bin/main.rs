use rand_distr::{Distribution, Poisson};

fn main() {
    let lambda = 1.0e306f64;
    let poisson = Poisson::<f64>::new(lambda).unwrap();
    let mut rng = rand::thread_rng();
    let k = poisson.sample(&mut rng);
    println!("{}", k);
}
