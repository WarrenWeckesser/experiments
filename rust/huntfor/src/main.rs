//
// huntfor
//
// Search files for a given regular expression.
//
// huntfor is inspired by the Python program grin, written by Robert Kern.
//

extern crate ansi_term;
extern crate glob;
extern crate regex;
//#[macro_use]
extern crate structopt;

use ansi_term::Colour::{Fixed, Green};
use glob::glob;
use regex::Regex;
use std::collections::VecDeque;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::PathBuf;
use std::process;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "huntfor", about = "Search files for a regular expression.")]
struct Opt {
    /// Show the filename at the start of each output line.
    #[structopt(long = "each")]
    each: bool,

    /// Igore case when matching letters.
    #[structopt(short = "i", long = "ignore-case")]
    ignorecase: bool,

    /// Number of lines of context to print before a match.
    #[structopt(short = "B", long = "before-context", default_value = "0")]
    before: usize,

    /// Number of lines of context to print after a match.
    #[structopt(short = "A", long = "after-context", default_value = "0")]
    after: usize,

    /// Disable the use of colors to highlight the strings that were found.
    #[structopt(long = "nocolor")]
    nocolor: bool,

    /// Numeric foreground color for highlighting strings that were found.
    /// Valid values are from 0 to 255.  See the description of the
    /// --background option for more details.
    #[structopt(long = "foreground", default_value = "0")]
    foreground: u8,

    /// Numeric background color for highlighting strings that were found.
    /// Valid values are from 0 to 255.  Values 0 to 7 are the "normal"
    /// (actually dark) colors: 0: black, 1: dark red, 2: dark green,
    /// 3: dark yellow, 4: dark blue, 5: dark purple, 6: dark cyan,
    /// 7: gray.  Values 8 to 15 are the lighter versions of those
    /// colors: 8: gray, 9: red, 10: green, 11: yellow, 12: blue,
    /// 13: purple, 14: cyan, 15: light gray.  White is 231.
    #[structopt(long = "background", default_value = "14")]
    background: u8,

    /// Regular expression to search for.
    pattern: String,

    /// The files to search. A glob pattern may be given.
    /// For example, use "**/*.rs" to recursively search all files with the
    /// extension .rs in the current directory and all subdirectories.
    files: Vec<String>,
}

fn makesep(s: &str, opt: &Opt) -> String {
    if opt.nocolor {
        s.to_string()
    } else {
        Green.paint(s).to_string()
    }
}

fn print_line(path: &PathBuf, sep1: &str, linecount: usize, sep2: &str, line: &str, opt: &Opt) {
    if opt.each {
        print!("{}{}", path.to_string_lossy(), sep1);
    }
    println!("{:5}{}{}", linecount, sep2, line);
}

fn huntfilefor(path: &PathBuf, re: &Regex, opt: &Opt) {
    let file = match File::open(&path) {
        Err(why) => {
            eprintln!("Failed to open {}: {}", path.display(), why);
            process::exit(2);
        }
        Ok(file) => file,
    };
    let mut beforelines: VecDeque<String> = VecDeque::new();

    let reader = BufReader::new(file);

    let mut first_match = true;
    let mut linecount = 0;
    let mut after_countdown: usize = 0;

    let sep = makesep("|", opt);
    let sep_before = makesep("-", opt);
    let sep_after = makesep("+", opt);

    // This is the replacement pattern used in the call to re.replace_all
    // below.  It wraps the named reference '$w' in an ANSI terminal color.
    let colorized = Fixed(opt.foreground)
        .on(Fixed(opt.background))
        .paint("$w")
        .to_string();

    for line in reader.lines() {
        linecount += 1;
        match line {
            Ok(mut line) => {
                if re.is_match(&line) {
                    if first_match & !opt.each {
                        // This is the first regex match in the file, and the
                        // command line flag --each was not given, so print the
                        // filename.
                        let name = path.to_string_lossy().to_string();
                        let out = if opt.nocolor {
                            name
                        } else {
                            Green.bold().paint(name).to_string()
                        };
                        println!("{}:", out);
                        first_match = false;
                    }
                    if opt.before > 0 {
                        let mut blen: usize = beforelines.len();
                        for beforeline in &beforelines {
                            print_line(path, &sep, linecount - blen, &sep_before, beforeline, opt);
                            blen -= 1;
                        }
                    }
                    if !opt.nocolor {
                        line = re.replace_all(&line, colorized.as_str()).to_string()
                    };
                    print_line(path, &sep, linecount, &sep, &line, opt);
                    after_countdown = opt.after;
                } else {
                    // Line does not contain the regex.
                    if after_countdown > 0 {
                        print_line(path, &sep, linecount, &sep_after, &line, opt);
                        after_countdown -= 1;
                    } else {
                        if opt.before > 0 {
                            if beforelines.len() == opt.before {
                                beforelines.pop_front();
                            }
                            beforelines.push_back(line);
                        }
                    }
                }
            }
            Err(_e) => (),
            //Err(e) => {
            //    eprintln!("ERROR: {}", e);
            //    eprintln!("Skipping file {}", path.display());
            //    break;
            //},
        }
    }
}

fn huntfor(opt: &Opt) {
    // Compile the regex provided by the user.  This compiled result
    // is not actually used, but it lets us detect errors and give an
    // error message that shows only the user's input.  If this is
    // successful, the regex will later be modifed and recompiled before
    // actually being used in the search.
    if let Some(e) = Regex::new(&opt.pattern).err() {
        eprintln!(
            "Failed to compile the regular expression. Error details:\n{}",
            e
        );
        process::exit(1)
    }

    let mut pattern = opt.pattern.clone();

    if opt.ignorecase {
        // Add the "case insensitive" flag to the pattern.
        pattern.insert_str(0, "(?i)")
    }

    // Wrap the pattern in '(?P<w>' and ')'.  This adds a named group around
    // the user's regex, with name w.
    pattern.insert_str(0, "(?P<w>");
    pattern.push_str(")");

    let re = Regex::new(&pattern).unwrap_or_else(|e| {
        eprintln!(
            "Failed to compile the regular expression. Error details:\n{}",
            e
        );
        process::exit(1)
    });

    for filepattern in &opt.files {
        match glob(filepattern) {
            Ok(files) => {
                for entry in files {
                    match entry {
                        Ok(path) => {
                            if path.is_file() {
                                huntfilefor(&path, &re, opt);
                            }
                        }
                        Err(e) => println!(
                            "Unexpected error while processing \"{}\".  Error details:\n{:?}",
                            filepattern, e
                        ),
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "There is something wrong with the file pattern \"{}\".  Error details:\n{}",
                    filepattern, e
                );
                process::exit(3);
            }
        }
    }
}

fn main() {
    let opt = Opt::from_args();
    huntfor(&opt);
}
