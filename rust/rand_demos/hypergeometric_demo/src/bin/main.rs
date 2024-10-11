use rand_distr::{Distribution, Hypergeometric};

fn get_rng(seed: u64) -> impl rand::Rng {
    const INC: u64 = 11634580027462260723;
    rand_pcg::Pcg32::new(seed, INC)
}

fn main() {
    let ntotal: u64 = 40;
    let ngood: u64 = 20;
    let ndraw: u64 = 19;
    let seed: u64 = 1236537458;
    //let nsamples: u64 = 25000000;
    let nsamples: u64 = 1;

    let hg = Hypergeometric::new(ntotal, ngood, ndraw).unwrap();
    let mut rng = get_rng(seed);

    println!("# N={} K={} n={}", ntotal, ngood, ndraw);
    for _ in 0..nsamples {
        let k = hg.sample(&mut rng);
        println!("{}", k);
    }
}
