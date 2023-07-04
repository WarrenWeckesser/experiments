use std::println;

fn compensated_sum_f32(x: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    for &val in x {
        sum += val as f64;
    }
    sum as f32
}

fn compensated_sum_f64(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for &val in x {
        let y = val - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

fn main() {
    let x = [
        1.0e-7f32, 1.0e-7f32, 1.0e-7f32, 1.0e-7f32, 1.0e-7f32, 1.0f32, -1.0f32,
    ];

    let xsum = compensated_sum_f32(&x);
    println!("xsum = {:e}", xsum);

    let y = [
        1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0f64, -1.0f64,
    ];

    let ysum = compensated_sum_f64(&y);
    println!("ysum = {:e}", ysum);
}
