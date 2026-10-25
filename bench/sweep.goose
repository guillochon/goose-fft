// Sweeps the column block size: the time of one transform per block size
// and length. For tuning, not a benchmark.
//
//   goose build bench/sweep.goose && bench/sweep

import std;
import os;
import fft from "..";

var sink = 0.0;

fn best<F>(reps: i64) -> f64 {
    var b = 1.0e30;
    for k in 7 {
        let t0 = clock();
        for i in reps { F(); }
        b = min(b, (clock() - t0) / reps);
    }
    return b;
}

fn main() {
    var r = rng { 1 };
    for n in [256, 1024, 4096, 16384, 65536, 262144, 1048576] {
        var x: fft::complex[>..] = [];
        for i in n { x.push(fft::complex { r.rand_flt(), r.rand_flt() }); }
        var line: u8[>..] = [];
        format(line, "n=", n, ":");
        for bp in [64, 128, 256, 512, 2048, 8192] {
            fft::BLOCK_CV4 = bp;
            let p = fft::plan_complex(n);
            let reps = max(1, 20000000 / (n * 20));
            let t = best(reps) { let y = fft::execute(p, x); sink += y[1].re; };
            format(line, "  ", bp, "=");
            format_flt(line, t * 1.0e6, 2);
        }
        print(line);
    }
    print(sink);
}
