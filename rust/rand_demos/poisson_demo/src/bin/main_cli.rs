use rand_distr::{Distribution, Poisson};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "poisson_sample",
    about = "Generate samples from the Poisson distribution"
)]
struct Opt {
    /// Poisson parameter
    #[structopt(short = "l")]
    l: f64,

    /// Number of samples to generate.
    #[structopt(short = "n", default_value = "1")]
    n: usize,

    /// RNG seed.
    #[structopt(short = "s", default_value = "1")]
    s: u64,
}

//
// This is from the value_stability test.
//
fn get_rng(seed: u64) -> impl rand::Rng {
    // For tests, we want a statistically good, fast, reproducible RNG.
    // PCG32 will do fine, and will be easy to embed if we ever need to.
    const INC: u64 = 11634580027462260723;
    rand_pcg::Pcg32::new(seed, INC)
}

fn main() {
    let opt = Opt::from_args();

    let poisson = Poisson::<f32>::new(opt.l as f32).unwrap();
    let mut rng = get_rng(opt.s);

    for _ in 0..opt.n {
        let k = poisson.sample(&mut rng);
        println!("{}", k);
    }
}
