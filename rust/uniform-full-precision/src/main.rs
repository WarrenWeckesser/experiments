use rand::distributions::Uniform;
use rand::Rng;
use std::println;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "uniform_full_precision",
    about = "Generate uniform f32 samples from the positive interval [a, b)."
)]
struct Opt {
    /// Lower bound (inclusive) of the interval. Must not be less than f32::MIN_POSITIIVE.
    #[structopt(short = "a")]
    a: f32,

    /// Upper bound (exclusive) of the interval.  Must be greater than `a`.
    #[structopt(short = "b")]
    b: f32,

    /// Number of samples to generate.
    #[structopt(short = "n", default_value = "1")]
    n: usize,

    /// RNG seed.
    #[structopt(short = "s", default_value = "1")]
    s: u64,
}

fn float_split(x: f32) -> (u32, u32) {
    let n = x.to_bits();
    let exponent = n >> (f32::MANTISSA_DIGITS - 1);
    let mantissa_mask = (1 << (f32::MANTISSA_DIGITS - 1)) - 1;
    let mantissa = n & mantissa_mask;
    (exponent, mantissa)
}

/// Full precision uniform sampler from the positive interval [a, b).
///
/// This is experimental, and has not been rigorously checked for
/// off-by-one errors!
///
/// Requires `a > f32::MIN_POSITIVE`, `b > a`, and ln_2(b/a) <= 41 (more or
/// less).
///
/// A few characteristics of this generator:
/// * The generator will never return the value `b`.
/// * All representable floating point (`f32`) values in the interval [a, b)
///   are possible outputs.
/// * The time to generate a sample is roughly proportional to ceil(ln_2(b/a)).
///
fn uniform_sample<R: Rng + ?Sized>(a: f32, b: f32, rng: &mut R) -> f32 {
    // TO DO:
    // * Clean up and simplify repeated code.
    // * Validate `a` and `b`.
    // * Extend to `a <= 0`. (Note that `a < b < 0`
    //   would be easy: negate `a` and `b`, compute the
    //   sample `x` and return `-x`.
    // * Make generic for `f32` and `f64`.
    // * Refactor to fit the `rand` crate style of generators.

    // In the following, an "octave" refers to the interval from a power
    // of 2 to the next power of 2.  E.g. [4, 8) is an octave; [0.5, 16) is
    // five octaves; the interval [1.5, 10.0) contains two full octaves
    // ([2, 4) and [4, 8)), plus intervals below and above the full octaves.

    let floats_per_octave = 1u64 << (f32::MANTISSA_DIGITS - 1);

    // eprintln!("floats_per_octave = {}", floats_per_octave);

    let (ae, am) = float_split(a);
    let (be, bm) = float_split(b);

    // eprintln!("ae, am = {} {}", ae, am);
    // eprintln!("be, bm = {} {}", be, bm);

    if (ae == be) | (((ae + 1) == be) & (bm == 0)) {
        // The interval is contained within one octave.
        let upper = if bm == 0 { floats_per_octave as u32 } else { bm };
        let u = Uniform::new(am, upper);
        let m = rng.sample(u);
        return
        f32::from_bits((ae << (f32::MANTISSA_DIGITS - 1)) + m)
            .try_into()
            .unwrap()
    }

    let num_low = if am > 0 { floats_per_octave - am as u64 } else { 0 };

    // eprintln!("num_low = {}", num_low);

    let ar = if am > 0 { ae + 1 } else { ae };

    // eprintln!("ar = {}", ar);

    let full_octaves = if ar < be { be - ar } else { 0 };

    // eprintln!("full_octaves = {}", full_octaves);

    let num_intervals = full_octaves + if am > 0 { 1 } else { 0 } + if bm > 0 { 1 } else { 0 };

    // eprintln!("num_intervals = {}", num_intervals);

    if num_intervals > 41 {
        panic!("Interval is too large.");
    }

    let mut wtotal = num_low;
    let start = if num_low > 0 { 1 } else { 0 };
    // eprintln!("start = {}", start);
    for k in start..(full_octaves + start) {
        // eprintln!("k = {}  1<<k = {}  wtotal = {} (before)", k, 1 << k, wtotal);
        wtotal += (1u64 << k) * floats_per_octave;
        // eprintln!("k = {}  wtotal = {} (after)", k, wtotal);
    }
    if bm > 0 {
        wtotal += (1u64 << (full_octaves + start)) * bm as u64;
    }

    let u = Uniform::new(0, wtotal);
    let v = rng.sample(u);

    let mut w = num_low;
    let mut interval = 0;
    if (num_low > 0) && (v < num_low) {
        interval = 0;
    } else {
        let start = if num_low > 0 { 1 } else { 0 };
        for k in start..(full_octaves + start) {
            w += (1 << k) * floats_per_octave;
            if v < w {
                interval = k;
                break;
            }
        }
        if v >= w {
            interval = full_octaves + start;
        }
    }
    let e = ae + interval;
    let sample_low = if interval == 0 && am > 0 { am } else { 0 };
    let sample_high = if interval == (num_intervals - 1) && bm > 0 {
        bm
    } else {
        floats_per_octave as u32
    };
    let u = Uniform::new(sample_low, sample_high);
    let m = rng.sample(u);
    f32::from_bits((e << (f32::MANTISSA_DIGITS - 1)) + m)
        .try_into()
        .unwrap()
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
    let mut rng = get_rng(opt.s);
    if opt.a < f32::MIN_POSITIVE {
        eprintln!("error: `a` must not be less than f32::MIN_POSITIVE ({:e})", f32::MIN_POSITIVE);
        std::process::exit(1);
    }
    if opt.b <= opt.a {
        eprintln!("error: `b` must be greater than `a`");
        std::process::exit(1);
    }

    // eprintln!("a = {}  b = {}", opt.a, opt.b);
    for _ in 0..opt.n {
        let x = uniform_sample(opt.a, opt.b, &mut rng);
        println!("{}", x);
    }
}
