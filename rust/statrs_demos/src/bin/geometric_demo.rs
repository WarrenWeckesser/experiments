use rand;
use rand::distributions::Distribution;
use statrs::distribution::{DiscreteCDF, Geometric};

fn out(x: f64, newline: bool) {
    let a = x.abs();
    if (a == 0.0) || ((a > 1e-6) && (a < 1e6)) {
        print!("{:<20.12}", x);
    } else {
        print!("{:<20.12e}", x);
    }
    if newline {
        println!();
    }
}

fn main() {
    let p1 = 1e-9f64;
    let p2 = 1e-17f64;
    let geom1 = Geometric::new(p1).unwrap();
    let geom2 = Geometric::new(p2).unwrap();
    println!("CDF values");
    println!("                          p");
    print!("  k  ");
    out(p1, false);
    print!("   ");
    out(p2, true);
    println!("------------------------------------------------");
    for k in 0..10 {
        print!("{:3}  ", k);
        out(geom1.cdf(k), false);
        print!("   ");
        out(geom2.cdf(k), true);
    }

    let mut r = rand::thread_rng();
    println!();
    println!("Random samples from Geometric with p={}", p1);
    for _ in 0..5 {
        let sample = geom1.sample(&mut r);
        println!("{}", sample);
    }
}
