use combiter::CombIterArray;

fn main() {
    let ci = CombIterArray::<3, 4>::new();
    for c in ci {
        println!("{:?}", c);
    }
}
