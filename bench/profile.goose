// Times the phases of one six-step transform, by running the same steps as
// fft::transform with a clock around each. For tuning, not a benchmark.
//
//   goose build bench/profile.goose && bench/profile [n]

import std;
import os;
import fft from "..";

var sink = 0.0;
var t_load_a = 0.0;
var t_run_a = 0.0;
var t_tt = 0.0;
var t_load_b = 0.0;
var t_run_b = 0.0;
var t_store_b = 0.0;

fn transform(p, x: fft::complex[:], y: fft::complex[:]) {
    var t0 = clock();
    fn lap(acc: f64&) { let u = clock(); acc += u - t0; t0 = u; }
    let n1 = p.n1;
    let n2 = p.n2;
    let g1 = p.g1;
    let g2 = p.g2;
    let zn = fft::zt_size(p);
    let cap = fft::block_cap(p);
    let bs = cap + fft::PAD;
    let work = p.data[p.work_off..p.work_off + fft::work_size(p)];
    let z = work[..zn];
    let a = work[zn + fft::PAD..zn + fft::PAD + cap];
    let t = work[zn + fft::PAD + bs..zn + fft::PAD + bs + cap];
    var c0 = 0;
    while c0 < g1 {
        let w = min(p.wa, g1 - c0);
        let size = w * n2;
        fft::load_a(x, false, n1, n2, c0, w, a[..size]);
        lap(t_load_a);
        let k = fft::run(p.sa, p.data, w, a[..size], t[..size]);
        lap(t_run_a);
        if k { fft::twiddle_transpose(p, c0, w, a[..size], z); }
        else { fft::twiddle_transpose(p, c0, w, t[..size], z); }
        lap(t_tt);
        c0 += w;
    }
    if p.wb == g2 {
        let k = fft::run(p.sb, p.data, g2, z, t[..zn]);
        lap(t_run_b);
        if k { fft::store_b(y, false, 1.0, n1, n2, 0, g2, z); }
        else { fft::store_b(y, false, 1.0, n1, n2, 0, g2, t[..zn]); }
        lap(t_store_b);
        return;
    }
    var d0 = 0;
    while d0 < g2 {
        let w = min(p.wb, g2 - d0);
        let size = w * n1;
        fft::load_b(z, g2, n1, d0, w, a[..size]);
        lap(t_load_b);
        let k = fft::run(p.sb, p.data, w, a[..size], t[..size]);
        lap(t_run_b);
        if k { fft::store_b(y, false, 1.0, n1, n2, d0, w, a[..size]); }
        else { fft::store_b(y, false, 1.0, n1, n2, d0, w, t[..size]); }
        lap(t_store_b);
        d0 += w;
    }
}

fn main() {
    let argv = args();
    var n = 4096;
    if argv.len > 1 { let v, ok = parse_int(argv[1]); n = v; }
    let p = fft::plan_complex(n);
    var radices: u8[>..] = [];
    for k in p.sa.count { format(radices, p.sa.radix[k], " "); }
    radices.append("| ");
    for k in p.sb.count { format(radices, p.sb.radix[k], " "); }
    print("n=", n, " n1=", p.n1, " n2=", p.n2, " g1=", p.g1, " g2=", p.g2, " wa=", p.wa, " wb=", p.wb,
          " radices ", radices);
    var r = rng { 1 };
    var x: fft::complex[>..] = [];
    for i in n { x.push(fft::complex { r.rand_flt(), r.rand_flt() }); }
    var y: fft::complex[>..] = [];
    y.resize(n, fft::complex { 0.0, 0.0 });
    let reps = max(10, 50000000 / (n * 20));
    let want = fft::execute(p, x);
    transform(p, x, y);
    var d = 0.0;
    for v, i in want { d = max(d, abs(v.re - y[i].re) + abs(v.im - y[i].im)); }
    print("max diff vs fft::execute ", d);
    t_load_a = 0.0; t_run_a = 0.0; t_tt = 0.0; t_load_b = 0.0; t_run_b = 0.0; t_store_b = 0.0;
    let t0 = clock();
    for i in reps { transform(p, x, y); sink += y[1].re; }
    let total = (clock() - t0) / reps;
    let lib0 = clock();
    for i in reps { fft::execute_into(p, x, y, false); sink += y[1].re; }
    let lib = (clock() - lib0) / reps;
    fn show(name: const u8[:], v: f64) {
        var line: u8[>..] = [];
        format(line, "  ", name, " ");
        format_flt(line, v / reps * 1.0e6, 3);
        format(line, " us");
        print(line);
    }
    print("total (timed) ", total * 1.0e6, " us; fft::execute_into ", lib * 1.0e6, " us");
    show("load A        ", t_load_a);
    show("passes A      ", t_run_a);
    show("twid+transpose", t_tt);
    show("load B        ", t_load_b);
    show("passes B      ", t_run_b);
    show("store B       ", t_store_b);
    print(sink);
}
