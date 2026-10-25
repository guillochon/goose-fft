// Times fft's transforms at several lengths, for comparison with
// bench_fftw.c (FFTW) and bench_numpy.py (NumPy and SciPy's pocketfft).
// Each timing is the best of several samples, each sample enough calls to
// take about a millisecond; the results go to stdout and to the CSV named on
// the command line (bench_goose.csv by default).
//
//   goose build bench/bench.goose && bench/bench [out.csv]
//
// Speeds are FFTW's convention: 5 n log2(n) / time for complex transforms,
// half that for real ones, whatever the length.

import std;
import os;
import math;
import fft from "..";

var sink = 0.0;       // results feed this, so no computation is dead
var csv: u8[>..] = [];

let POW2 = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576];
let MIXED = [100, 1000, 3000, 10000, 100000, 1000000];
let PRIME = [1009, 10007];

fn flops(kind: const u8[:], n: i64) -> f64 {
    let f = 5.0 * n * log2(n * 1.0);
    if kind == "rfft" { f / 2.0 } else { f }
}

// The best time per call of F over samples of `reps` calls each: at least 5
// samples and 0.2 s.
fn time_it<F>(kind: const u8[:], n: i64) {
    var reps = 1;
    loop {                                           // calls per sample: about 1 ms
        let t0 = clock();
        for i in reps { F(); }
        if clock() - t0 > 0.001 || reps > 1000000 { break; }
        reps *= 2;
    }
    var best = 1.0e30;
    var total = 0.0;
    var samples = 0;
    while samples < 5 || total < 0.2 {
        let t0 = clock();
        for i in reps { F(); }
        let dt = clock() - t0;
        best = min(best, dt / reps);
        total += dt;
        samples++;
    }
    format(csv, kind, ",", n, ",", best, "\n");
    var line: u8[>..] = [];
    format(line, kind, " n=", n, ": ");
    format_flt(line, best * 1.0e6, 3);
    format(line, " us  ");
    format_flt(line, flops(kind, n) / best / 1.0e6, 0);
    format(line, " MFLOPS");
    print(line);
}

fn bench_complex(n: i64, r: rng&) {
    var x: fft::complex[>..] = [];
    for i in n { x.push(fft::complex { 2.0 * r.rand_flt() - 1.0, 2.0 * r.rand_flt() - 1.0 }); }
    let p = fft::plan_complex(n);
    // Into a preallocated output, as FFTW is timed.
    var y: fft::complex[>..] = [];
    y.resize(n, fft::complex { 0.0, 0.0 });
    time_it("fft", n) { fft::execute_into(p, x, y, false); sink += y[0].re; };
}

fn main() {
    csv.append("kind,n,seconds\n");
    var r = rng { 42 };
    for n in POW2 { bench_complex(n, r); }
    for n in MIXED { bench_complex(n, r); }
    for n in PRIME { bench_complex(n, r); }
    let argv = args();
    let path = if argv.len > 1 { argv[1] } else { "bench_goose.csv" };
    if !write_file(path, csv) { abort(str("cannot write ", path)); }
    print("checksum ", sink);
}
