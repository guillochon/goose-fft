// Tests for the fft package.
//
//   goose test/test_fft.goose                                (in-process)
//   goose build test/test_fft.goose && test/test_fft          (native)
//
// Each check that fails prints its group and what it checked; the run ends
// with a summary and exits 1 if anything failed. Transforms are checked
// against a direct O(n^2) DFT, whose twiddles are reduced mod n in integers,
// at every length up to 160 and at larger lengths of every kind (powers of
// two, smooth, primes, Bluestein); at lengths too long for that, against
// direct sums at sampled bins. Then the identities a transform must satisfy
// (inverse, linearity, Parseval, shifts, real symmetry), the real and 2-D
// transforms against the complex one, and the helpers against hand-worked
// values.

import std;
import os;
import math;
import fft from "..";

var passed = 0;
var failed = 0;
var group_name: u8[>..] = [];

fn group(name: const u8[:]) {
    group_name.clear();
    group_name.append(name);
}

fn check(ok: bool, what: const u8[:]) {
    if ok { passed++; return; }
    failed++;
    print("FAIL [", group_name, "] ", what);
}

fn check_close(got: f64, want: f64, tol: f64, what: const u8[:]) {
    if abs(got - want) <= tol * max(1.0, abs(want)) { passed++; return; }
    failed++;
    print("FAIL [", group_name, "] ", what, ": got ", got, ", want ", want);
}

// max |got - want| <= tol * max(1, max |want|).
fn check_cvec(got: fft::complex[:], want: fft::complex[:], tol: f64, what: const u8[:]) {
    guard got.len == want.len else {
        failed++;
        print("FAIL [", group_name, "] ", what, ": length ", got.len, ", want ", want.len);
        return;
    }
    var d = 0.0;
    var big = 1.0;
    for w, i in want {
        d = max(d, max(abs(got[i].re - w.re), abs(got[i].im - w.im)));
        big = max(big, max(abs(w.re), abs(w.im)));
    }
    if d <= tol * big { passed++; return; }
    failed++;
    print("FAIL [", group_name, "] ", what, ": max abs diff ", d, " > ", tol * big);
}

fn check_rvec(got: f64[:], want: f64[:], tol: f64, what: const u8[:]) {
    guard got.len == want.len else {
        failed++;
        print("FAIL [", group_name, "] ", what, ": length ", got.len, ", want ", want.len);
        return;
    }
    var d = 0.0;
    var big = 1.0;
    for w, i in want { d = max(d, abs(got[i] - w)); big = max(big, abs(w)); }
    if d <= tol * big { passed++; return; }
    failed++;
    print("FAIL [", group_name, "] ", what, ": max abs diff ", d, " > ", tol * big);
}

fn random_cvec(n: i64, r: rng&) -> fft::complex[>..] {
    var out: fft::complex[>..] = [];
    for i in n { out.push(fft::complex { 2.0 * r.rand_flt() - 1.0, 2.0 * r.rand_flt() - 1.0 }); }
    return out;
}

fn random_rvec(n: i64, r: rng&) -> f64[>..] {
    var out: f64[>..] = [];
    for i in n { out.push(2.0 * r.rand_flt() - 1.0); }
    return out;
}

// One bin of the direct DFT, sign -1 forward or +1 backward (unnormalized).
fn dft_bin(x: fft::complex[:], k: i64, sign: f64) -> fft::complex {
    let n = x.len;
    var sr = 0.0;
    var si = 0.0;
    for v, j in x {
        let t = TAU * ((j * k) % n) / n;
        let c = cos(t);
        let s = sign * sin(t);
        sr += v.re * c - v.im * s;
        si += v.re * s + v.im * c;
    }
    return fft::complex { sr, si };
}

fn dft(x: fft::complex[:], sign: f64) -> fft::complex[>..] {
    var out: fft::complex[>..] = [];
    for k in x.len { out.push(dft_bin(x, k, sign)); }
    return out;
}

// The error bound for a length-n transform of values of size about 1: the
// FFT's own error grows as log n, the direct DFT's as sqrt n.
fn tol_for(n: i64) -> f64 { 4.0e-15 * (1.0 + log2(n * 1.0)) * sqrt(n * 1.0) }

fn test_against_dft(n: i64, r: rng&) {
    let x = random_cvec(n, r);
    let want = dft(x, -1.0);
    let got = fft::fft(x);
    check_cvec(got, want, tol_for(n), str("fft n=", n));
    let back = fft::ifft(got);
    check_cvec(back, x, 4.0e-15 * (1.0 + log2(n * 1.0)), str("ifft(fft(x)) n=", n));
}

fn test_real(n: i64, r: rng&) {
    let x = random_rvec(n, r);
    var xc: fft::complex[>..] = [];
    for v in x { xc.push(fft::complex { v, 0.0 }); }
    let full = fft::fft(xc);
    let half = fft::rfft(x);
    check_cvec(half, full[..n / 2 + 1], 1.0e-14 * (1.0 + log2(n * 1.0)), str("rfft n=", n));
    let back = fft::irfft(half, n);
    check_rvec(back, x, 1.0e-14 * (1.0 + log2(n * 1.0)), str("irfft(rfft(x)) n=", n));
}

// Large lengths: sampled bins against direct sums, and the round trip.
fn test_large(n: i64, r: rng&) {
    let x = random_cvec(n, r);
    let p = fft::plan_complex(n);
    let y = fft::execute(p, x);
    var worst = 0.0;
    for t in 12 {
        let k = if t < 4 { t } else if t < 8 { n - 1 - (t - 4) } else { r.rand_int(n) };
        let want = dft_bin(x, k, -1.0);
        worst = max(worst, max(abs(y[k].re - want.re), abs(y[k].im - want.im)));
    }
    // A bin of n random terms is about sqrt(n) in size; its error is a few
    // ulp of that times log n.
    check(worst <= 2.0e-15 * sqrt(n * 1.0) * (1.0 + log2(n * 1.0)) * 4.0,
          str("sampled bins n=", n, " (worst ", worst, ")"));
    let back = fft::execute_inverse(p, y);
    check_cvec(back, x, 1.0e-14 * (1.0 + log2(n * 1.0)), str("round trip n=", n));
}

fn main() {
    var r = rng { 12345 };

    group("small lengths against the DFT");
    for n in 1..161 { test_against_dft(n, r); }

    group("larger lengths against the DFT");
    for n in [256, 243, 625, 1000, 1024, 720, 2310, 4096, 210, 343, 961, 4 * 7 * 29, 31 * 31, 2048, 3000] {
        test_against_dft(n, r);
    }

    group("Bluestein lengths against the DFT");
    for n in [37, 41, 67, 101, 127, 257, 509, 1009, 2 * 37 * 41, 3 * 97, 4 * 131, 997 * 2] { test_against_dft(n, r); }

    group("large lengths, sampled");
    for n in [16384, 65536, 131072, 262144, 1048576, 100000, 1000000, 3 * 65536, 10007, 65537] { test_large(n, r); }

    group("exact transforms");
    {
        // A unit impulse transforms to all ones; a constant to n at 0.
        var x: fft::complex[>..] = [];
        for i in 12 { x.push(fft::complex { if i == 0 { 1.0 } else { 0.0 }, 0.0 }); }
        let y = fft::fft(x);
        var ones: fft::complex[>..] = [];
        for i in 12 { ones.push(fft::complex { 1.0, 0.0 }); }
        check_cvec(y, ones, 1.0e-15, "impulse");
        let z = fft::fft(ones);
        var twelve: fft::complex[>..] = [];
        for i in 12 { twelve.push(fft::complex { if i == 0 { 12.0 } else { 0.0 }, 0.0 }); }
        check_cvec(z, twelve, 1.0e-15, "constant");
        // A pure tone at bin 3 of 64: everything at 3.
        var tone: fft::complex[>..] = [];
        for j in 64 { let t = TAU * 3 * j / 64; tone.push(fft::complex { cos(t), sin(t) }); }
        let tt = fft::fft(tone);
        var want: fft::complex[>..] = [];
        for k in 64 { want.push(fft::complex { if k == 3 { 64.0 } else { 0.0 }, 0.0 }); }
        check_cvec(tt, want, 1.0e-14, "tone at bin 3");
    }

    group("identities");
    for n in [8, 30, 64, 100, 128, 210, 512, 1000, 1009, 4096] {
        let x = random_cvec(n, r);
        let y = random_cvec(n, r);
        let fx = fft::fft(x);
        let fy = fft::fft(y);
        // Linearity: fft(2x + y) = 2 fft(x) + fft(y).
        var s: fft::complex[>..] = [];
        for v, i in x { s.push(fft::complex { 2.0 * v.re + y[i].re, 2.0 * v.im + y[i].im }); }
        var want: fft::complex[>..] = [];
        for v, i in fx { want.push(fft::complex { 2.0 * v.re + fy[i].re, 2.0 * v.im + fy[i].im }); }
        check_cvec(fft::fft(s), want, 1.0e-14 * (1.0 + log2(n * 1.0)), str("linearity n=", n));
        // Parseval: sum |x|^2 = sum |X|^2 / n.
        var ex = 0.0;
        var ef = 0.0;
        for v in x { ex += v.re * v.re + v.im * v.im; }
        for v in fx { ef += v.re * v.re + v.im * v.im; }
        check_close(ef / n, ex, 1.0e-13, str("Parseval n=", n));
        // Shift: x[j - 1] (cyclic) transforms to X[k] w^k.
        var sh: fft::complex[>..] = [];
        for j in n { sh.push(x[(j + n - 1) % n]); }
        var ws: fft::complex[>..] = [];
        for v, k in fx {
            let t = TAU * k / n;
            let c = cos(t);
            let sn = 0.0 - sin(t);
            ws.push(fft::complex { v.re * c - v.im * sn, v.re * sn + v.im * c });
        }
        check_cvec(fft::fft(sh), ws, 1.0e-14 * (1.0 + log2(n * 1.0)), str("shift n=", n));
    }

    group("plans and in place");
    for n in [16, 100, 1024, 1009, 4096, 65536] {
        let p = fft::plan_complex(n);
        let x = random_cvec(n, r);
        let want = fft::fft(x);
        // The same plan twice, and in place.
        check_cvec(fft::execute(p, x), want, 1.0e-15, str("plan reused n=", n));
        var y: fft::complex[>..] = [];
        y.append(x);
        fft::execute_into(p, y, y, false);
        check_cvec(y, want, 1.0e-15, str("in place n=", n));
        fft::execute_into(p, y, y, true);
        check_cvec(y, x, 1.0e-14 * (1.0 + log2(n * 1.0)), str("in place inverse n=", n));
    }

    group("real transforms");
    for n in 1..70 { test_real(n, r); }
    for n in [100, 128, 256, 1000, 1024, 1026, 2 * 1009, 1009, 4096, 4098, 65536, 100000, 2 * 65537] { test_real(n, r); }
    {
        // Real input has a conjugate-symmetric spectrum: rfft is its first half.
        let x = random_rvec(64, r);
        let h = fft::rfft(x);
        check(abs(h[0].im) == 0.0 && abs(h[32].im) == 0.0, "rfft: DC and Nyquist bins are real");
        let p = fft::plan_real(1000);
        let a = random_rvec(1000, r);
        let s1 = fft::execute_real(p, a);
        let s2 = fft::execute_real(p, a);
        check_cvec(s1, s2, 0.0, "real plan reused");
    }

    group("2-D");
    for dims in [[1, 1], [1, 8], [8, 1], [4, 4], [3, 5], [16, 12], [7, 33], [32, 64]] {
        let rows = dims[0];
        let cols = dims[1];
        let x = random_cvec(rows * cols, r);
        // The 2-D DFT, directly: each output a double sum.
        var want: fft::complex[>..] = [];
        for kr in rows {
            for kc in cols {
                var sr = 0.0;
                var si = 0.0;
                for jr in rows {
                    for jc in cols {
                        let t = TAU * (((jr * kr) % rows) * 1.0 / rows + ((jc * kc) % cols) * 1.0 / cols);
                        let v = x[jr * cols + jc];
                        sr += v.re * cos(t) + v.im * sin(t);
                        si += v.im * cos(t) - v.re * sin(t);
                    }
                }
                want.push(fft::complex { sr, si });
            }
        }
        let got = fft::fft2(x, rows, cols);
        check_cvec(got, want, 1.0e-13, str("fft2 ", rows, "x", cols));
        check_cvec(fft::ifft2(got, rows, cols), x, 1.0e-14, str("ifft2(fft2) ", rows, "x", cols));
    }

    group("helpers");
    {
        check_rvec(fft::fftfreq(5, 1.0), [0.0, 0.2, 0.4, -0.4, -0.2], 1.0e-15, "fftfreq(5)");
        check_rvec(fft::fftfreq(6, 0.5), [0.0, 1.0 / 3.0, 2.0 / 3.0, -1.0, -2.0 / 3.0, -1.0 / 3.0], 1.0e-15, "fftfreq(6, 0.5)");
        check_rvec(fft::rfftfreq(6, 1.0), [0.0, 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0], 1.0e-15, "rfftfreq(6)");
        check_rvec(fft::rfftfreq(5, 1.0), [0.0, 0.2, 0.4], 1.0e-15, "rfftfreq(5)");
        let a: f64[] = [0.0, 1.0, 2.0, 3.0, 4.0];
        check_rvec(fft::fftshift(a), [3.0, 4.0, 0.0, 1.0, 2.0], 0.0, "fftshift odd");
        check_rvec(fft::ifftshift(fft::fftshift(a)), a, 0.0, "ifftshift odd");
        let b: f64[] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        check_rvec(fft::fftshift(b), [3.0, 4.0, 5.0, 0.0, 1.0, 2.0], 0.0, "fftshift even");
        check_rvec(fft::ifftshift(fft::fftshift(b)), b, 0.0, "ifftshift even");
        check_rvec(fft::convolve([1.0, 2.0, 3.0], [0.0, 1.0, 0.5]), [0.0, 1.0, 2.5, 4.0, 1.5], 1.0e-14, "convolve small");
        // Against the direct sum, at awkward lengths.
        for lens in [[1, 1], [7, 3], [100, 37], [1000, 999], [4096, 5]] {
            let u = random_rvec(lens[0], r);
            let v = random_rvec(lens[1], r);
            var want: f64[>..] = [];
            want.resize(u.len + v.len - 1, 0.0);
            for a, i in u { for b, j in v { want[i + j] += a * b; } }
            check_rvec(fft::convolve(u, v), want, 1.0e-13, str("convolve ", lens[0], " by ", lens[1]));
        }
    }

    print(passed, " passed, ", failed, " failed");
    if failed > 0 { exit(1); }
}
