// A quick check of a few lengths, best of several runs: for comparing
// variants while tuning, not a benchmark.
//
//   goose build bench/quick.goose && bench/quick

import std;
import os;
import fft from "..";

var sink = 0.0;

fn main() {
    var r = rng { 3 };
    var line: u8[>..] = [];
    for n in [256, 1024, 4096, 65536, 1000, 100000] {
        var x: fft::complex[>..] = [];
        for i in n { x.push(fft::complex { r.rand_flt(), r.rand_flt() }); }
        var y: fft::complex[>..] = [];
        y.resize(n, fft::complex { 0.0, 0.0 });
        let p = fft::plan_complex(n);
        let reps = max(20, 40000000 / (n * 17));
        var best = 1.0e30;
        for k in 15 {
            let t0 = clock();
            for i in reps { fft::execute_into(p, x, y, false); sink += y[1].re; }
            best = min(best, (clock() - t0) / reps);
        }
        format(line, "n=", n, " ");
        format_flt(line, best * 1.0e6, 3);
        format(line, " us   ");
    }
    print(line);
    print(sink);
}
