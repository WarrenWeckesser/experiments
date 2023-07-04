use num_traits::float::Float;
use num_traits::NumAssignOps;
use std::println;

fn compensated_sum<T>(x: &[T]) -> T
where
    T: Float + NumAssignOps,
{
    let mut sum = T::zero();
    let mut c = T::zero();
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

    let xsum = compensated_sum(&x);
    println!("xsum = {:e}", xsum);

    let y = [
        1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0f64, -1.0f64,
    ];

    let ysum = compensated_sum(&y);
    println!("ysum = {:e}", ysum);

    let v = vec![
        1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0f64, -1.0f64,
    ];

    let vsum = compensated_sum(&v);
    println!("vsum = {:e}", vsum);
}
