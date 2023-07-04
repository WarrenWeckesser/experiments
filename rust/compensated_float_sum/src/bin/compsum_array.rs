use std::println;

trait CompensatedSummable<T> {
    fn compensated_sum(&self) -> T;
}

impl CompensatedSummable<f32> for &[f32; 5] {
    fn compensated_sum(&self) -> f32 {
        // Use an f64 accumulator
        let mut sum = 0.0f64;
        for &val in *self {
            sum += val as f64;
        }
        sum as f32
    }
}

impl CompensatedSummable<f64> for &[f64; 5] {
    fn compensated_sum(&self) -> f64 {
        // Kahan compensated sum
        let mut sum = 0.0;
        let mut c = 0.0;
        for &val in *self {
            let y = val - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum
    }
}

fn main() {
    let x = [1.0e-7f32, 1.0e-7f32, 1.0e-7f32, 1.0f32, -1.0f32];

    let xsum = (&x).compensated_sum();
    println!("xsum = {:e}", xsum);

    let y = [1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0f64, -1.0f64];

    let ysum = (&y).compensated_sum();
    println!("ysum = {:e}", ysum);
}
