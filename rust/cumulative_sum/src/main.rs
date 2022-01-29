fn main() {
    let v = vec![0.1, 2.4, 10.0, -1.5, 10.0];
    let s: Vec<f64> = v
        .iter()
        .scan(0.0, |sum, x| {
            *sum += x;
            Some(*sum)
        })
        .collect();
    let rs: Vec<f64> = v
        .iter()
        .rev()
        .scan(0.0, |sum, x| {
            *sum += x;
            Some(*sum)
        })
        .collect();
    println!("             v: {:?}", v);
    println!("cumulative sum: {:?}", s);
    println!("rev & cum. sum: {:?}", rs);
    let rrs: Vec<f64> = rs.into_iter().rev().collect();
    println!(" rev, sum, rev: {:?}", rrs);
    let mut t: Vec<f64> = v
        .iter()
        .rev()
        .scan(0.0, |sum, x| {
            *sum += x;
            Some(*sum)
        })
        .collect();
    t.reverse();
    println!(" rev, sum, rev: {:?}", t);
}
