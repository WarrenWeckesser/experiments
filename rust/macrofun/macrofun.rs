// Experiments with keyword function parameters and default values.

fn increment(x: u32, dx: u32, d2x: u32) -> u32 {
    x + dx + d2x * d2x
}

macro_rules! incr {
    ($var:ident) => {
        increment($var, 0);
    };
    ($var:ident, dx = $dxval:expr) => {
        increment($var, $dxval, 0);
    };
    ($var:ident, d2x = $d2xval:expr) => {
        increment($var, 0, $d2xval);
    };
    ($var:ident, dx = $dxval:expr, d2x = $d2xval:expr) => {
        increment($var, $dxval, $d2xval);
    };
    ($var:ident, d2x = $d2xval:expr, dx = $dxval:expr) => {
        increment($var, $dxval, $d2xval);
    };
}

struct IncrementArgs {
    x: u32,
    dx: u32,
    d2x: u32,
}

macro_rules! inc {
    ( $( $var:ident = $value:expr ),* ) => {
        {
            static DEFAULT_ARGS: IncrementArgs = IncrementArgs{x: 0, dx: 0, d2x: 0};
            let args: IncrementArgs =
                IncrementArgs{$($var: $value,)* ..DEFAULT_ARGS};
            increment(args.x, args.dx, args.d2x)
        }
    };
}

// Define a function whose arguments have default values.
// All the arguments must be given default values.
// Defines a regular function call $name, and a macro called $name!.
// In the macro, keywords must be used; it will not accept positional
// arguments.
// XXX This does not work with generics,  e.g. fn foo<T>(...).

macro_rules! fndef {
    (fn $name:ident ( $( $var:ident : $t:ty = $val:expr),* ) -> $ret:ty $body:block) => {
        fn $name($($var:$t),*) -> $ret $body
        struct Args {
            $($var:$t),*
        }
        impl Default for Args {
            fn default() -> Args {
                Args {
                    $(
                    $var: $val
                    ),*
                }
            }
        }
        macro_rules! $name {
            // Can't nest repetitions, so we have separate patterns for one
            // argument, two arguments, three arguments, ...
            ($var1:ident = $val1:expr) => {
                {
                    let args: Args = Args{$var1: $val1, .. Default::default()};
                    $name($(args.$var),*)
                }
            };
            ($var1:ident = $val1:expr, $var2:ident = $val2:expr) => {
                {
                    let args: Args = Args {
                                         $var1: $val1,
                                         $var2: $val2,
                                         .. Default::default()
                                     };
                    $name($(args.$var),*)
                }
            };
            ($var1:ident = $val1:expr,
             $var2:ident = $val2:expr,
             $var3:ident = $val3:expr) => {
                {
                    let args: Args = Args {
                                         $var1: $val1,
                                         $var2: $val2,
                                         $var3: $val3,
                                         .. Default::default()
                                     };
                    $name($(args.$var),*)
                }
            };
            ($var1:ident = $val1:expr,
             $var2:ident = $val2:expr,
             $var3:ident = $val3:expr,
             $var4:ident = $val4:expr) => {
                {
                    let args: Args = Args {
                                         $var1: $val1,
                                         $var2: $val2,
                                         $var3: $val3,
                                         $var4: $val4,
                                         .. Default::default()
                                     };
                    $name($(args.$var),*)
                }
            };
            ($var1:ident = $val1:expr,
             $var2:ident = $val2:expr,
             $var3:ident = $val3:expr,
             $var4:ident = $val4:expr,
             $var5:ident = $val5:expr) => {
                {
                    let args: Args = Args {
                                         $var1: $val1,
                                         $var2: $val2,
                                         $var3: $val3,
                                         $var4: $val4,
                                         $var5: $val5,
                                         .. Default::default()
                                     };
                    $name($(args.$var),*)
                }
            };
        }
    }
}

fndef!(
    fn axplusb(a: u32 = 1, x: u32 = 0, b: u32 = 0) -> u32 {
        a*x + b
    }
);

fn main() {
    let x = 12;
    println!("x = {}", x);
    println!("{}", increment(x, 0, 2));
    println!("{}", incr!(x, dx = 2));
    println!("{}", incr!(x, d2x = 2));
    println!("{}", incr!(x, dx = 3, d2x = 2));
    println!("{}", incr!(x, d2x = 3, dx = 2));
    println!("");
    println!("{}", inc!(x = x, d2x = 2, dx = 2 * x));
    println!("{}", inc!(x = x, d2x = 2, dx = 5));

    let y = axplusb(1, x, 10);
    println!("y = {}", y);

    let y = axplusb!(b = 10, x = x);
    println!("y = {}", y);
}
