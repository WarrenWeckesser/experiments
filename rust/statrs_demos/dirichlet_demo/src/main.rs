use rand;
use rand::distributions::Distribution;
use statrs::distribution::Dirichlet;


fn main() {
    let n = 1000;
    println!("Checking {} Dirichlet samples", n);

    let dirichlet = Dirichlet::new(vec![0.001, 0.001, 0.001]).unwrap();
    let mut r = rand::thread_rng();
    let mut nancount = 0;
    for _ in 0..n {
        let sample = dirichlet.sample(&mut r);
        if sample.iter().any(|x| x.is_nan()) {
            nancount += 1;
        }
    }
    println!("nancount: {}", nancount);
}
