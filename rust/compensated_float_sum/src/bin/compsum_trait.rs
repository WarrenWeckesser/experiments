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

trait CompensatedSummable<T> {
    fn compensated_sum(&self) -> T;
}

impl CompensatedSummable<f32> for &Vec<f32> {
    fn compensated_sum(&self) -> f32 {
        compensated_sum_f32(self)
    }
}

impl CompensatedSummable<f32> for [f32] {
    fn compensated_sum(&self) -> f32 {
        compensated_sum_f32(self)
    }
}

impl CompensatedSummable<f64> for &Vec<f64> {
    fn compensated_sum(&self) -> f64 {
        compensated_sum_f64(self)
    }
}

impl CompensatedSummable<f64> for [f64] {
    fn compensated_sum(&self) -> f64 {
        compensated_sum_f64(self)
    }
}

fn main() {
    let xv = vec![
        1.0e-7f32, 1.0e-7f32, 1.0e-7f32, 1.0e-7f32, 1.0e-7f32, 1.0f32, -1.0f32,
    ];

    let sumxv = compensated_sum_f32(&xv);
    println!("sumxv = {:e}", sumxv);

    let xvsum = (&xv).compensated_sum();
    println!("xvsum = {:e}", xvsum);

    let xa = [
        1.0e-7f32, 1.0e-7f32, 1.0e-7f32, 1.0e-7f32, 1.0e-7f32, 1.0f32, -1.0f32,
    ];

    let sumxa = compensated_sum_f32(&xa);
    println!("sumxa = {:e}", sumxa);

    let xasum = (&xa).compensated_sum();
    println!("xasum = {:e}", xasum);

    let yv = vec![
        1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0f64, -1.0f64,
    ];

    let sumyv = compensated_sum_f64(&yv);
    println!("sumyv = {:e}", sumyv);

    let yvsum = (&yv).compensated_sum();
    println!("yvsum = {:e}", yvsum);

    let ya = [
        1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0e-16f64, 1.0f64, -1.0f64,
    ];

    let sumya = compensated_sum_f64(&ya);
    println!("sumya = {:e}", sumya);

    let yasum = (&ya).compensated_sum();
    println!("yasum = {:e}", yasum);
}
