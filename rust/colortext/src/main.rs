extern crate ansi_term;
extern crate regex;

use ansi_term::Colour::Green;
use regex::Regex;

fn main() {
    let re = Regex::new(r"(?P<w>w.*?d)").unwrap();
    let s = "Hello, world! Where in the world are you?";
    let replacement = Green.bold().paint("$w").to_string();
    let colorized = re.replace_all(s, replacement.as_str());

    println!("{}", colorized);
}
