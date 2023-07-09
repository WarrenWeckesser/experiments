#[macro_use]
extern crate approx; // For the macro assert_relative_eq!
extern crate nalgebra as na;
use na::{Rotation3, Vector3};

fn main() {
    let axis = Vector3::x_axis();
    println!("axis = ({}, {}, {})", axis.x, axis.y, axis.z);

    let angle = 1.57;
    let b = Rotation3::from_axis_angle(&axis, angle);

    let b_angles = b.euler_angles();
    println!(
        "b.euler_angles() = ({}, {}, {})",
        b_angles.0, b_angles.1, b_angles.2
    );

    assert_relative_eq!(b.axis().unwrap(), axis);
    assert_relative_eq!(b.angle(), angle);
}
