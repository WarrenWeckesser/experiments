use rand_distr::{Distribution, NegativeBinomial};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "negative_binomial_sample",
    about = "Generate samples from the negative binomial distribution"
)]
struct Opt {
    /// Number of successes required before stopping.
    #[structopt(short = "r")]
    r: f64,

    /// Probability of success.
    #[structopt(short = "p")]
    p: f64,

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

    let nb = NegativeBinomial::<f32>::new(opt.r as f32, opt.p as f32).unwrap();
    let mut rng = get_rng(opt.s);

    for _ in 0..opt.n {
        let k = nb.sample(&mut rng);
        println!("{}", k);
    }
}
