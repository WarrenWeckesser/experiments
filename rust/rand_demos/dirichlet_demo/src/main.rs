use rand::distributions::Distribution;
use rand_distr::Dirichlet;

fn main() {
    let n = 1000;
    println!("Checking {} Dirichlet samples", n);

    let dirichlet = Dirichlet::new(&[0.001, 0.001, 0.001]).unwrap();
    let mut r = rand::thread_rng();
    let mut nancount = 0;
    for _ in 0..n {
        let sample: Vec<f64> = dirichlet.sample(&mut r);
        if sample.iter().any(|x| x.is_nan()) {
            nancount += 1;
        }
    }
    println!("nancount: {}", nancount);
}
