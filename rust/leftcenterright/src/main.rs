// Left-center-right dice game simulation.
// Copyright (c) 2021 Warren Weckesser

use rand::Rng;
use std::cmp;
use std::thread;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "leftcenterright", about = "Simulate left-center-right game.")]
struct Opt {
    /// Number of players.
    #[structopt(short = "p", long = "players")]
    players: usize,

    /// Starting number of chips.
    #[structopt(short = "s", long = "start", default_value = "3")]
    start: usize,

    /// Number of games to simulate.
    #[structopt(short = "g", long = "games", default_value = "1000")]
    games: usize,

    // Number of parallel jobs (i.e. threads) to use; 0 implies serial.
    #[structopt(short = "j", long = "jobs", default_value = "0")]
    jobs: usize,
}

fn next(i: usize, n: usize) -> usize {
    if (i + 1) == n {
        0
    } else {
        i + 1
    }
}

fn prev(i: usize, n: usize) -> usize {
    if i == 0 {
        n - 1
    } else {
        i - 1
    }
}

fn count_nonzero(v: &[usize]) -> usize {
    v.iter().filter(|&n| *n != 0).count()
}

fn sim(players: usize, start: usize) -> usize {
    let mut chips = vec![start; players];
    let mut turn: usize = 0;
    let mut rng = rand::thread_rng();
    while count_nonzero(&chips) > 1 {
        for _n in 0..cmp::min(chips[turn], 3) {
            let roll = rng.gen_range(0..6);
            // 0 means left, 1 means center, 2 means right.
            match roll {
                0 => {
                    chips[turn] -= 1;
                    chips[prev(turn, players)] += 1;
                }
                1 => {
                    chips[turn] -= 1;
                }
                2 => {
                    chips[turn] -= 1;
                    chips[next(turn, players)] += 1;
                }
                _ => {}
            }
        }
        turn = next(turn, players);
    }
    // There is one nonzero entry in chips; return its index.
    chips.iter().position(|&n| n > 0).unwrap()
}

fn stats(players: usize, start: usize, games: usize) -> Vec<usize> {
    let mut wins = vec![0; players];
    for _n in 0..games {
        let winner = sim(players, start);
        wins[winner] += 1;
    }
    wins
}

fn parallel_stats(opt: &Opt) -> Vec<usize> {
    // (b, r) = divmod(opt.games, opt.jobs)
    let b = opt.games / opt.jobs;
    let r = opt.games % opt.jobs;
    let batchsizes: Vec<usize> = (0..opt.jobs).map(|k| b + (k < r) as usize).collect();

    let mut children = vec![];

    for i in 0..opt.jobs {
        let players = opt.players;
        let start = opt.start;
        let batchsize = batchsizes[i];
        children.push(thread::spawn(move || stats(players, start, batchsize)));
    }

    let mut wins = vec![0; opt.players];
    for child in children {
        let batch_wins = child.join().unwrap();
        for i in 0..opt.players {
            wins[i] += batch_wins[i];
        }
    }
    wins
}

fn main() {
    let opt = Opt::from_args();
    println!("opt.players : {}", opt.players);
    println!("opt.start   : {}", opt.start);
    println!("opt.jobs    : {}", opt.jobs);
    println!("opt.games   : {}", opt.games);
    let wins = if opt.jobs == 0 {
        stats(opt.players, opt.start, opt.games)
    } else {
        parallel_stats(&opt)
    };
    println!("{:?}", wins);
}
