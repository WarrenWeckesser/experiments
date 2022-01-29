// Example of the use of the quantile method.

use statrs::statistics::Data;
use statrs::statistics::OrderStatistics;

fn main() {
    let data = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0, 1.0, 6.0];
    let mut data = Data::new(data);
    for q in [0.01, 0.83, 0.99] {
        println!("{}: {}", q, data.quantile(q));
    }
}
