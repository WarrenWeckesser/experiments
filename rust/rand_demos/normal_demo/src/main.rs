use rand_distr::{Distribution, Normal};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "normal_demo",
    about = "Generate samples from the normal distribution"
)]
struct Opt {
    /// Mean
    #[structopt(short = "m", long = "mu", default_value = "0")]
    mu: f64,

    /// Standard deviation
    #[structopt(short = "s", long = "sigma", default_value = "1")]
    sigma: f64,

    /// Number of samples to generate.
    #[structopt(short = "n", default_value = "1")]
    n: usize,

    /// Random number generator seed.
    #[structopt(short = "r", default_value = "1")]
    r: u64,
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

    let distr = Normal::<f32>::new(opt.mu as f32, opt.sigma as f32).unwrap();
    let mut rng = get_rng(opt.r);

    for _ in 0..opt.n {
        let k = distr.sample(&mut rng);
        println!("{}", k);
    }
}
