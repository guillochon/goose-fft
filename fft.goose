// fft: fast Fourier transforms in pure Goose.
//
// Transforms of any length: mixed radix (8, 4, 2, 3, 5 and any small odd
// prime) for lengths whose prime factors are small, Bluestein's algorithm
// for the rest. Complex values are `fft::complex { re, im }`, so an array of
// them is interleaved like C99's `double complex` or NumPy's complex128.
//
// The conventions are NumPy's: the forward transform is unnormalized,
// X[k] = sum_j x[j] exp(-2 pi i j k / n), and the inverse divides by n.
//
// Everything is in namespace fft: `fft::fft(x)`. A `plan` holds the
// twiddle factors and scratch for one length; the plan-less functions build
// one per call, so reuse a plan when transforming many arrays of the same
// length.
//
// How it is fast. A transform of n = n1 * n2 points runs as Bailey's six
// steps: the n1 columns of x, seen as n2 rows by n1 columns, are transformed
// (n2 points each), multiplied by twiddle factors and transposed, and the
// n2 columns of that are transformed (n1 points each). Every step works on
// four columns at once, one per lane of a `v4`, a struct of four f64s that
// the language adds and multiplies elementwise. A `cv4` holds the real and
// imaginary v4s of four complex numbers side by side, and each butterfly is
// a few whole-v4 operations on cv4s, which the C compiler turns into AVX2
// instructions as written. Lengths that don't split into multiples of four
// leave lanes idle rather than going slow.

import std;
import math;
import codelets;

namespace fft;

struct complex { re: f64, im: f64 }

// Four lanes, one column each: the unit every pass computes on.
struct v4 { a: f64, b: f64, c: f64, d: f64 }

// Four complex numbers, real parts then imaginary parts.
struct cv4 { re: v4, im: v4 }

fn splat(x: f64) -> v4 { v4 { x, x, x, x } }
fn csplat(re: f64, im: f64) -> cv4 { cv4 { splat(re), splat(im) } }

// v times t, complex, lane by lane.
inline fn twid(v: cv4, t: cv4) -> cv4 { cv4 { v.re * t.re - v.im * t.im, v.re * t.im + v.im * t.re } }

// ---------------------------------------------------------------------------
// Plans

let MAX_STAGES = 64;

// Prime factors up to this are radices of their own; a length with a larger
// one goes through Bluestein's algorithm.
let MAX_RADIX = 31;

// A column block holds about this many cv4s per array (64 bytes each, two
// arrays of them), so it stays in cache.
var BLOCK_CV4 = 256;

let KIND_COMPLEX = 0;
let KIND_REAL = 1;

// The Stockham passes for one length: radices in order, and where each
// pass's twiddles start in the plan's data.
struct stages { len: i64, count: i64, radix: i64[MAX_STAGES], tw: i64[MAX_STAGES] }

// A plan for one transform length. The core is a complex transform of
// length `len` = n1 * n2 (n for a complex plan, or the padded length of
// Bluestein's convolution), in g1 = ceil(n1 / 4) groups of columns for step
// A and g2 = ceil(n2 / 4) for step B, wa and wb groups to a block.
struct plan {
    n: i64,
    kind: i64,
    cn: i64,                    // the complex transform's length: n, or n/2 for a real plan of even n
    len: i64,
    n1: i64,
    n2: i64,
    g1: i64,
    g2: i64,
    wa: i64,
    wb: i64,
    sa: stages,                 // n2 points, down the columns of x
    sb: stages,                 // n1 points, down the columns of the transpose
    codelets: bool,             // n1 and n2 both have codelets: each step is one sweep
    mid_off: i64,               // w^(j1 k2) for the four j1 of group g at [g * n2 + k2]
    blue: bool,                 // Bluestein: chirp and the filter's transform, four complex to a cv4
    chirp_off: i64,
    filt_off: i64,
    real_off: i64,              // a real plan's twiddles w^k, k = 0..n/4, four to a cv4
    realv_off: i64,             // the same from k = 1, four consecutive k to a cv4 for the vector loops
    work_off: i64,              // scratch for transform: zt, then two block buffers
    data: cv4[>..],
}

// exp(-2 pi i k / n) for any integer k, accurate to the last bit or two:
// k is reduced mod n first, and the angle is taken in the first octant.
fn root(k: i64, n: i64) -> f64, f64 {
    // The angle 2 pi j / n is quadrant q plus pi r / (2n), r in [0, n), all
    // in integers; past the octant, take the complement from the quadrant's end.
    let j = k % n;
    let q = 4 * j / n;
    let r = 4 * j - q * n;
    var c = 0.0;
    var s = 0.0;
    if 2 * r <= n {
        let t = PI * r / (2 * n);
        c = cos(t); s = sin(t);
    } else {
        let t = PI * (n - r) / (2 * n);
        c = sin(t); s = cos(t);
    }
    // Rotate by q quarter turns.
    var rc = c;
    var rs = s;
    if q == 1 { rc = 0.0 - s; rs = c; }
    else if q == 2 { rc = 0.0 - c; rs = 0.0 - s; }
    else if q == 3 { rc = s; rs = 0.0 - c; }
    return rc, 0.0 - rs;
}

// The radices of n: eights, with a four or two (or a pair of fours) for the
// rest of the power of two, then odd primes in increasing order; false when
// a prime factor is larger than MAX_RADIX.
fn factor(n: i64, radix: i64[MAX_STAGES]&, count: i64&) -> bool {
    var m = n;
    var e = 0;
    while m % 2 == 0 { m /= 2; e++; }
    count = 0;
    var eights = e / 3;
    var rest = e % 3;
    if rest == 1 && eights > 0 { eights--; rest = 4; }
    for i in eights { radix[count] = 8; count++; }
    if rest == 4 { radix[count] = 4; count++; radix[count] = 4; count++; }
    else if rest == 2 { radix[count] = 4; count++; }
    else if rest == 1 { radix[count] = 2; count++; }
    var f = 3;
    while m > 1 {
        if f > MAX_RADIX { return false; }
        while m % f == 0 { radix[count] = f; count++; m /= f; }
        f += 2;
    }
    return true;
}

fn factorable(n: i64) -> bool {
    var radix: i64[MAX_STAGES] = [0; MAX_STAGES];
    var count = 0;
    return factor(n, radix, count);
}

// The smallest 2^a 3^b 5^c at least n, and a multiple of 16.
fn smooth_at_least(n: i64) -> i64 {
    var best = next_pow2(max(n, 16));
    var p5 = 1;
    while p5 < best {
        var p35 = p5;
        while p35 < best {
            var v = p35 * 16;
            while v < n { v *= 2; }
            best = min(best, v);
            p35 *= 3;
        }
        p5 *= 5;
    }
    return best;
}

// The Stockham passes for len, their twiddles appended to data, every lane
// the same: per pass, w^(t*j) at [(r-1)*j + t-1] for t = 1..r-1, and for a
// radix past 5 its r x r DFT matrix after them.
fn make_stages(data: cv4[>..]&, len: i64) -> stages {
    var st = stages { len: len, count: 0, radix: [0; MAX_STAGES], tw: [0; MAX_STAGES] };
    let ok = factor(len, st.radix, st.count);
    assert(ok);
    var span = len;                    // the length the pass transforms
    for k in st.count {
        let r = st.radix[k];
        let m = span / r;
        st.tw[k] = data.len;
        for j in m { for t in 1..r { let c, s = root(t * j, span); data.push(csplat(c, s)); } }
        if r > 5 && r != 8 {
            for t in r { for u in r { let c, s = root(t * u, r); data.push(csplat(c, s)); } }
        }
        span = m;
    }
    return st;
}

// The most even split n1 * n2 = len with n1 <= n2.
fn split_even(len: i64) -> i64 {
    var n1 = 1;
    var d = 1;
    while d * d <= len {
        if len % d == 0 { n1 = d; }
        d++;
    }
    return n1;
}

// The split n1 * n2 = len with n1 <= n2 both with codelets, most even, or 0.
fn split_codelets(len: i64) -> i64 {
    var n1 = 0;
    var d = 1;
    while d * d <= len {
        if len % d == 0 && has_codelet(d) && has_codelet(len / d) { n1 = d; }
        d++;
    }
    return n1;
}

// Sets up the core transform of length len.
fn setup_core(p: plan&, len: i64) {
    p.len = len;
    var n1 = split_codelets(len);
    p.codelets = n1 > 0;
    if !p.codelets { n1 = split_even(len); }
    let n2 = len / n1;
    p.n1 = n1;
    p.n2 = n2;
    p.g1 = (n1 + 3) / 4;
    p.g2 = (n2 + 3) / 4;
    // Blocks small enough to stay in L1 and L2 suit lengths that fit in
    // cache; past those, wider blocks read memory in longer runs.
    let bsize = if len <= 65536 { BLOCK_CV4 } else { min(8192, len / 128) };
    p.wa = min(p.g1, max(1, bsize / n2));
    p.wb = min(p.g2, max(1, bsize / n1));
    p.sa = make_stages(p.data, n2);
    p.sb = make_stages(p.data, n1);
    p.mid_off = p.data.len;
    for g in p.g1 { for k2 in n2 { p.data.push(mid_twiddle(n1, len, g, k2)); } }
    // The scratch transform works in, kept with the plan so that a transform
    // allocates nothing.
    p.work_off = p.data.len;
    p.data.push_n(csplat(0.0, 0.0), work_size(p));
}

// zt, then two block buffers big enough for either step, PAD cv4s apart:
// arrays a power of two long, back to back, would put a pass's loads at the
// same address mod 4096 as the stores before them, and the core stalls each
// such load until it knows the store went elsewhere.
let PAD = 9;

fn zt_size(p) -> i64 { p.g2 * p.n1 }
fn block_cap(p) -> i64 { max(p.wa * p.n2, p.wb * p.n1) }
fn work_size(p) -> i64 { zt_size(p) + PAD + 2 * (block_cap(p) + PAD) }

// w^(j1 k2) for the four columns j1 of group g.
fn mid_twiddle(n1: i64, len: i64, g: i64, k2: i64) -> cv4 {
    var re: f64[4] = [0.0; 4];
    var im: f64[4] = [0.0; 4];
    for l in 4 {
        let j1 = 4 * g + l;
        if j1 < n1 { let c, s = root(j1 * k2, len); re[l] = c; im[l] = s; }
    }
    return cv4 { v4 { re[0], re[1], re[2], re[3] }, v4 { im[0], im[1], im[2], im[3] } };
}

fn new_plan(n: i64, kind: i64) -> plan {
    let none = stages { len: 0, count: 0, radix: [0; MAX_STAGES], tw: [0; MAX_STAGES] };
    return plan { n: n, kind: kind, cn: n, len: n, n1: 0, n2: 0, g1: 0, g2: 0, wa: 0, wb: 0, sa: none, sb: none,
                  codelets: false, mid_off: 0, blue: false, chirp_off: 0, filt_off: 0, real_off: 0, realv_off: 0, work_off: 0, data: [] };
}

// Complex numbers stored four to a cv4 in a plan's data.
fn put_complex(data: cv4[>..]&, x: complex[:]) {
    var k = 0;
    while k < x.len {
        var re: f64[4] = [0.0; 4];
        var im: f64[4] = [0.0; 4];
        for l in 4 { if k + l < x.len { re[l] = x[k + l].re; im[l] = x[k + l].im; } }
        data.push(cv4 { v4 { re[0], re[1], re[2], re[3] }, v4 { im[0], im[1], im[2], im[3] } });
        k += 4;
    }
}

fn get_complex(data: cv4[:], off: i64, k: i64) -> complex {
    let v = data[off + k / 4];
    return match k % 4 {
        0 => complex { v.re.a, v.im.a },
        1 => complex { v.re.b, v.im.b },
        2 => complex { v.re.c, v.im.c },
        _ => complex { v.re.d, v.im.d },
    };
}

// A plan for complex transforms of length n.
fn plan_complex(n: i64) -> plan {
    assert(n >= 1);
    var p = new_plan(n, KIND_COMPLEX);
    setup_complex(p, n);
    return p;
}

// The complex transform of length cn for p: mixed radix, or Bluestein's
// algorithm on a padded length when cn has a large prime factor.
fn setup_complex(p: plan&, cn: i64) {
    p.cn = cn;
    if factorable(cn) {
        setup_core(p, cn);
        return;
    }
    // Bluestein: a circular convolution of length m >= 2cn - 1.
    let m = smooth_at_least(2 * cn - 1);
    p.blue = true;
    setup_core(p, m);
    // The chirp exp(-pi i k^2 / cn) = root(k^2 mod 2cn, 2cn).
    var chirp: complex[>..] = [];
    for k in cn { let c, s = root((k * k) % (2 * cn), 2 * cn); chirp.push(complex { c, s }); }
    p.chirp_off = p.data.len;
    put_complex(p.data, chirp);
    // The filter: conj(chirp) at 0..cn and wrapped to m-cn+1..m, transformed
    // and divided by m (the inverse transform's normalization, taken here once).
    var b: complex[>..] = [];
    b.resize(m, complex { 0.0, 0.0 });
    for c, k in chirp {
        b[k] = complex { c.re / m, (0.0 - c.im) / m };
        if k > 0 { b[m - k] = b[k]; }
    }
    var f: complex[>..] = [];
    f.resize(m, complex { 0.0, 0.0 });
    transform(p, b, f, false, 1.0);
    p.filt_off = p.data.len;
    put_complex(p.data, f);
}

// ---------------------------------------------------------------------------
// Butterflies: the r-point DFT of r cv4s, lane by lane.

struct c2 { o0: cv4, o1: cv4 }
struct c3 { o0: cv4, o1: cv4, o2: cv4 }
struct c4 { o0: cv4, o1: cv4, o2: cv4, o3: cv4 }
struct c5 { o0: cv4, o1: cv4, o2: cv4, o3: cv4, o4: cv4 }
struct c8 { o0: cv4, o1: cv4, o2: cv4, o3: cv4, o4: cv4, o5: cv4, o6: cv4, o7: cv4 }

inline fn dft2(a0: cv4, a1: cv4) -> c2 {
    c2 { cv4 { a0.re + a1.re, a0.im + a1.im }, cv4 { a0.re - a1.re, a0.im - a1.im } }
}

inline fn dft4(a0: cv4, a1: cv4, a2: cv4, a3: cv4) -> c4 {
    let b0r = a0.re + a2.re;
    let b0i = a0.im + a2.im;
    let b1r = a0.re - a2.re;
    let b1i = a0.im - a2.im;
    let b2r = a1.re + a3.re;
    let b2i = a1.im + a3.im;
    let b3r = a1.im - a3.im;            // -i (a1 - a3)
    let b3i = a3.re - a1.re;
    return c4 { cv4 { b0r + b2r, b0i + b2i }, cv4 { b1r + b3r, b1i + b3i },
                cv4 { b0r - b2r, b0i - b2i }, cv4 { b1r - b3r, b1i - b3i } };
}

let SQRT_HALF = 0.7071067811865476;

inline fn dft8(a0: cv4, a1: cv4, a2: cv4, a3: cv4, a4: cv4, a5: cv4, a6: cv4, a7: cv4) -> c8 {
    let r2 = splat(SQRT_HALF);
    // Radix 2 across the halves, the odd half times w8^k.
    let b0r = a0.re + a4.re;
    let b0i = a0.im + a4.im;
    let b1r = a1.re + a5.re;
    let b1i = a1.im + a5.im;
    let b2r = a2.re + a6.re;
    let b2i = a2.im + a6.im;
    let b3r = a3.re + a7.re;
    let b3i = a3.im + a7.im;
    let d0r = a0.re - a4.re;
    let d0i = a0.im - a4.im;
    let e1r = a1.re - a5.re;
    let e1i = a1.im - a5.im;
    let d1r = r2 * (e1r + e1i);         // (1 - i) / sqrt 2
    let d1i = r2 * (e1i - e1r);
    let d2r = a2.im - a6.im;            // -i
    let d2i = a6.re - a2.re;
    let e3r = a3.re - a7.re;
    let e3i = a3.im - a7.im;
    let d3r = r2 * (e3i - e3r);         // (-1 - i) / sqrt 2
    let d3i = splat(0.0) - r2 * (e3r + e3i);
    // Two 4-point DFTs: the even outputs from b, the odd from d.
    let f0r = b0r + b2r;
    let f0i = b0i + b2i;
    let f1r = b0r - b2r;
    let f1i = b0i - b2i;
    let f2r = b1r + b3r;
    let f2i = b1i + b3i;
    let f3r = b1i - b3i;
    let f3i = b3r - b1r;
    let g0r = d0r + d2r;
    let g0i = d0i + d2i;
    let g1r = d0r - d2r;
    let g1i = d0i - d2i;
    let g2r = d1r + d3r;
    let g2i = d1i + d3i;
    let g3r = d1i - d3i;
    let g3i = d3r - d1r;
    return c8 { cv4 { f0r + f2r, f0i + f2i }, cv4 { g0r + g2r, g0i + g2i },
                cv4 { f1r + f3r, f1i + f3i }, cv4 { g1r + g3r, g1i + g3i },
                cv4 { f0r - f2r, f0i - f2i }, cv4 { g0r - g2r, g0i - g2i },
                cv4 { f1r - f3r, f1i - f3i }, cv4 { g1r - g3r, g1i - g3i } };
}

let SIN60 = 0.8660254037844386;

inline fn dft3(a0: cv4, a1: cv4, a2: cv4) -> c3 {
    let half = splat(0.5);
    let sin60 = splat(SIN60);
    let t1r = a1.re + a2.re;
    let t1i = a1.im + a2.im;
    let t2r = a0.re - half * t1r;
    let t2i = a0.im - half * t1i;
    let t3r = sin60 * (a1.re - a2.re);
    let t3i = sin60 * (a1.im - a2.im);
    return c3 { cv4 { a0.re + t1r, a0.im + t1i },
                cv4 { t2r + t3i, t2i - t3r },          // t2 - i t3
                cv4 { t2r - t3i, t2i + t3r } };        // t2 + i t3
}

let C5_1 = 0.30901699437494745;      // cos(2 pi / 5)
let C5_2 = -0.8090169943749475;      // cos(4 pi / 5)
let S5_1 = 0.9510565162951535;       // sin(2 pi / 5)
let S5_2 = 0.5877852522924731;       // sin(4 pi / 5)

inline fn dft5(a0: cv4, a1: cv4, a2: cv4, a3: cv4, a4: cv4) -> c5 {
    let c51 = splat(C5_1);
    let c52 = splat(C5_2);
    let s51 = splat(S5_1);
    let s52 = splat(S5_2);
    let t1r = a1.re + a4.re;
    let t1i = a1.im + a4.im;
    let t2r = a2.re + a3.re;
    let t2i = a2.im + a3.im;
    let t3r = a1.re - a4.re;
    let t3i = a1.im - a4.im;
    let t4r = a2.re - a3.re;
    let t4i = a2.im - a3.im;
    let m1r = a0.re + c51 * t1r + c52 * t2r;
    let m1i = a0.im + c51 * t1i + c52 * t2i;
    let m2r = a0.re + c52 * t1r + c51 * t2r;
    let m2i = a0.im + c52 * t1i + c51 * t2i;
    let n1r = s51 * t3r + s52 * t4r;
    let n1i = s51 * t3i + s52 * t4i;
    let n2r = s52 * t3r - s51 * t4r;
    let n2i = s52 * t3i - s51 * t4i;
    return c5 { cv4 { a0.re + t1r + t2r, a0.im + t1i + t2i },
                cv4 { m1r + n1i, m1i - n1r },          // m1 - i n1
                cv4 { m2r + n2i, m2i - n2r },          // m2 - i n2
                cv4 { m2r - n2i, m2i + n2r },          // m2 + i n2
                cv4 { m1r - n1i, m1i + n1r } };        // m1 + i n1
}

// ---------------------------------------------------------------------------
// The Stockham passes.
//
// A pass of radix r over a span of length r*m with stride s reads the r
// inputs x[q + s*(j + k*m)] for k < r and writes y[q + s*(r*j + t)] for t < r,
// the r-point DFT of the inputs times w^(t*j). Every index is a cv4, four
// columns transformed side by side; the operations on whole cv4s are what
// vectorize, so the loop over q may be short. Each input stream k and output
// stream t is sliced once per pass, so the loops index them at base + q,
// which the compiler proves in bounds. At j = 0 every twiddle is 1.

simd fn pass2(s: i64, m: i64, x: cv4[:], y: cv4[:], w: cv4[:]) {
    let l = s * m;
    assert(x.len == 2 * l && y.len == x.len && w.len == m);
    let x0 = x[..l];
    let x1 = x[l..2 * l];
    let ol = 2 * l - s;
    let y0 = y[..ol];
    let y1 = y[s..s + ol];
    for j in m {
        let bi = s * j;
        let bo = 2 * s * j;
        assert(bi + s <= x0.len && bi + s <= x1.len && bo + s <= y0.len && bo + s <= y1.len);
        if j == 0 {
            for q in s {
                let o = dft2(x0[bi + q], x1[bi + q]);
                y0[bo + q] = o.o0;
                y1[bo + q] = o.o1;
            }
        } else {
            let w1 = w[j];
            for q in s {
                let o = dft2(x0[bi + q], x1[bi + q]);
                y0[bo + q] = o.o0;
                y1[bo + q] = twid(o.o1, w1);
            }
        }
    }
}

simd fn pass4(s: i64, m: i64, x: cv4[:], y: cv4[:], w: cv4[:]) {
    let l = s * m;
    assert(x.len == 4 * l && y.len == x.len && w.len == 3 * m);
    let x0 = x[..l];
    let x1 = x[l..2 * l];
    let x2 = x[2 * l..3 * l];
    let x3 = x[3 * l..4 * l];
    let ol = 4 * l - 3 * s;
    let y0 = y[..ol];
    let y1 = y[s..s + ol];
    let y2 = y[2 * s..2 * s + ol];
    let y3 = y[3 * s..3 * s + ol];
    for j in m {
        let bi = s * j;
        let bo = 4 * s * j;
        assert(bi + s <= x0.len && bi + s <= x1.len && bi + s <= x2.len && bi + s <= x3.len);
        assert(bo + s <= y0.len && bo + s <= y1.len && bo + s <= y2.len && bo + s <= y3.len);
        if j == 0 {
            for q in s {
                let o = dft4(x0[bi + q], x1[bi + q], x2[bi + q], x3[bi + q]);
                y0[bo + q] = o.o0;
                y1[bo + q] = o.o1;
                y2[bo + q] = o.o2;
                y3[bo + q] = o.o3;
            }
        } else {
            let w1 = w[3 * j];
            let w2 = w[3 * j + 1];
            let w3 = w[3 * j + 2];
            for q in s {
                let o = dft4(x0[bi + q], x1[bi + q], x2[bi + q], x3[bi + q]);
                y0[bo + q] = o.o0;
                y1[bo + q] = twid(o.o1, w1);
                y2[bo + q] = twid(o.o2, w2);
                y3[bo + q] = twid(o.o3, w3);
            }
        }
    }
}

simd fn pass8(s: i64, m: i64, x: cv4[:], y: cv4[:], w: cv4[:]) {
    let l = s * m;
    assert(x.len == 8 * l && y.len == x.len && w.len == 7 * m);
    let x0 = x[..l];
    let x1 = x[l..2 * l];
    let x2 = x[2 * l..3 * l];
    let x3 = x[3 * l..4 * l];
    let x4 = x[4 * l..5 * l];
    let x5 = x[5 * l..6 * l];
    let x6 = x[6 * l..7 * l];
    let x7 = x[7 * l..8 * l];
    let ol = 8 * l - 7 * s;
    let y0 = y[..ol];
    let y1 = y[s..s + ol];
    let y2 = y[2 * s..2 * s + ol];
    let y3 = y[3 * s..3 * s + ol];
    let y4 = y[4 * s..4 * s + ol];
    let y5 = y[5 * s..5 * s + ol];
    let y6 = y[6 * s..6 * s + ol];
    let y7 = y[7 * s..7 * s + ol];
    for j in m {
        let bi = s * j;
        let bo = 8 * s * j;
        assert(bi + s <= x0.len && bi + s <= x1.len && bi + s <= x2.len && bi + s <= x3.len);
        assert(bi + s <= x4.len && bi + s <= x5.len && bi + s <= x6.len && bi + s <= x7.len);
        assert(bo + s <= y0.len && bo + s <= y1.len && bo + s <= y2.len && bo + s <= y3.len);
        assert(bo + s <= y4.len && bo + s <= y5.len && bo + s <= y6.len && bo + s <= y7.len);
        if j == 0 {
            for q in s {
                let o = dft8(x0[bi + q], x1[bi + q], x2[bi + q], x3[bi + q],
                             x4[bi + q], x5[bi + q], x6[bi + q], x7[bi + q]);
                y0[bo + q] = o.o0;
                y1[bo + q] = o.o1;
                y2[bo + q] = o.o2;
                y3[bo + q] = o.o3;
                y4[bo + q] = o.o4;
                y5[bo + q] = o.o5;
                y6[bo + q] = o.o6;
                y7[bo + q] = o.o7;
            }
        } else {
            let tw = w[7 * j..7 * j + 7];
            assert(tw.len == 7);
            for q in s {
                let o = dft8(x0[bi + q], x1[bi + q], x2[bi + q], x3[bi + q],
                             x4[bi + q], x5[bi + q], x6[bi + q], x7[bi + q]);
                y0[bo + q] = o.o0;
                y1[bo + q] = twid(o.o1, tw[0]);
                y2[bo + q] = twid(o.o2, tw[1]);
                y3[bo + q] = twid(o.o3, tw[2]);
                y4[bo + q] = twid(o.o4, tw[3]);
                y5[bo + q] = twid(o.o5, tw[4]);
                y6[bo + q] = twid(o.o6, tw[5]);
                y7[bo + q] = twid(o.o7, tw[6]);
            }
        }
    }
}

simd fn pass3(s: i64, m: i64, x: cv4[:], y: cv4[:], w: cv4[:]) {
    let l = s * m;
    assert(x.len == 3 * l && y.len == x.len && w.len == 2 * m);
    let x0 = x[..l];
    let x1 = x[l..2 * l];
    let x2 = x[2 * l..3 * l];
    let ol = 3 * l - 2 * s;
    let y0 = y[..ol];
    let y1 = y[s..s + ol];
    let y2 = y[2 * s..2 * s + ol];
    for j in m {
        let bi = s * j;
        let bo = 3 * s * j;
        assert(bi + s <= x0.len && bi + s <= x1.len && bi + s <= x2.len);
        assert(bo + s <= y0.len && bo + s <= y1.len && bo + s <= y2.len);
        let w1 = w[2 * j];
        let w2 = w[2 * j + 1];
        for q in s {
            let o = dft3(x0[bi + q], x1[bi + q], x2[bi + q]);
            y0[bo + q] = o.o0;
            y1[bo + q] = twid(o.o1, w1);
            y2[bo + q] = twid(o.o2, w2);
        }
    }
}

simd fn pass5(s: i64, m: i64, x: cv4[:], y: cv4[:], w: cv4[:]) {
    let l = s * m;
    assert(x.len == 5 * l && y.len == x.len && w.len == 4 * m);
    let x0 = x[..l];
    let x1 = x[l..2 * l];
    let x2 = x[2 * l..3 * l];
    let x3 = x[3 * l..4 * l];
    let x4 = x[4 * l..5 * l];
    let ol = 5 * l - 4 * s;
    let y0 = y[..ol];
    let y1 = y[s..s + ol];
    let y2 = y[2 * s..2 * s + ol];
    let y3 = y[3 * s..3 * s + ol];
    let y4 = y[4 * s..4 * s + ol];
    for j in m {
        let bi = s * j;
        let bo = 5 * s * j;
        assert(bi + s <= x0.len && bi + s <= x1.len && bi + s <= x2.len && bi + s <= x3.len && bi + s <= x4.len);
        assert(bo + s <= y0.len && bo + s <= y1.len && bo + s <= y2.len && bo + s <= y3.len && bo + s <= y4.len);
        let w1 = w[4 * j];
        let w2 = w[4 * j + 1];
        let w3 = w[4 * j + 2];
        let w4 = w[4 * j + 3];
        for q in s {
            let o = dft5(x0[bi + q], x1[bi + q], x2[bi + q], x3[bi + q], x4[bi + q]);
            y0[bo + q] = o.o0;
            y1[bo + q] = twid(o.o1, w1);
            y2[bo + q] = twid(o.o2, w2);
            y3[bo + q] = twid(o.o3, w3);
            y4[bo + q] = twid(o.o4, w4);
        }
    }
}

// Any odd radix, as a direct r-point DFT through its matrix (r x r): r^2
// work per r outputs.
simd fn pass_generic(r: i64, s: i64, m: i64, x: cv4[:], y: cv4[:], w: cv4[:], dm: cv4[:]) {
    assert(x.len == r * m * s && y.len == x.len && w.len == (r - 1) * m && dm.len == r * r);
    for j in m {
        for t in r {
            let o = s * (r * j + t);
            let yt = y[o..o + s];
            assert(yt.len == s);
            let z = csplat(0.0, 0.0);
            for q in s { yt[q] = z; }
            for k in r {
                let c = dm[t * r + k];
                let ak = x[s * (j + k * m)..s * (j + k * m) + s];
                assert(ak.len == s);
                for q in s {
                    let a = ak[q];
                    let v = yt[q];
                    yt[q] = cv4 { v.re + a.re * c.re - a.im * c.im, v.im + a.re * c.im + a.im * c.re };
                }
            }
            if t > 0 {
                let wt = w[(r - 1) * j + t - 1];
                for q in s { yt[q] = twid(yt[q], wt); }
            }
        }
    }
}

fn pass(st, k: i64, data: cv4[:], s: i64, m: i64, x: cv4[:], y: cv4[:]) {
    let r = st.radix[k];
    let o = st.tw[k];
    let w = data[o..o + (r - 1) * m];
    if r == 8 { pass8(s, m, x, y, w); }
    else if r == 4 { pass4(s, m, x, y, w); }
    else if r == 2 { pass2(s, m, x, y, w); }
    else if r == 3 { pass3(s, m, x, y, w); }
    else if r == 5 { pass5(s, m, x, y, w); }
    else {
        let d = o + (r - 1) * m;
        pass_generic(r, s, m, x, y, w, data[d..d + r * r]);
    }
}

// Runs the passes of st on s0 groups of four transforms at once: point j of
// group q is at q + s0*j in x, of length s0 * st.len. t is scratch of the
// same length. True if the result ends in x, false if in t.
fn run(st, data: cv4[:], s0: i64, x: cv4[:], t: cv4[:]) -> bool {
    var s = s0;
    var span = st.len;
    var in_x = true;
    for k in st.count {
        let r = st.radix[k];
        let m = span / r;
        if in_x { pass(st, k, data, s, m, x, t); }
        else { pass(st, k, data, s, m, t, x); }
        in_x = !in_x;
        s *= r;
        span = m;
    }
    return in_x;
}

// ---------------------------------------------------------------------------
// The six steps.
//
// Step A loads groups c0..c0+w of x's columns (n2 rows by n1 columns, four
// columns to a group) into a block, point j2 of group q at q + w*j2, runs the
// n2-point passes on it, and multiplies by the twiddles and transposes it
// into zt: the n1 rows of z = x's transformed columns, as cv4s of four
// consecutive k2, row j1's group gk at gk + g2*j1. Step B runs the n1-point
// passes on blocks of zt's groups and stores them to y: lane l of group gk at
// point k1 is y[4*gk + l + n2*k1]. Swapping real and imaginary parts on the
// way in and out makes the forward transform the inverse.

fn load4(e: complex[:], swap: bool) -> cv4 {
    let re = v4 { e[0].re, e[1].re, e[2].re, e[3].re };
    let im = v4 { e[0].im, e[1].im, e[2].im, e[3].im };
    return if swap { cv4 { im, re } } else { cv4 { re, im } };
}

inline fn load_a_in(x: complex[:], swap: bool, n1: i64, n2: i64, c0: i64, w: i64, a: cv4[:]) {
    assert(a.len == w * n2 && x.len == n1 * n2);
    // Full groups: four complex numbers to a cv4.
    let full = max(0, min(w, n1 / 4 - c0));
    for j2 in n2 {
        let row = x[n1 * j2..n1 * j2 + n1];
        let d = a[w * j2..w * j2 + w];
        assert(row.len == n1 && d.len == w && 4 * (c0 + full) <= n1);
        for q in full {
            let e = row[4 * (c0 + q)..4 * (c0 + q) + 4];
            assert(e.len == 4);
            d[q] = load4(e, swap);
        }
        // The group past the last full one: its missing columns are zero.
        for q in full..w {
            let c = 4 * (c0 + q);
            var e: complex[4] = [complex { 0.0, 0.0 }; 4];
            for l in 4 { if c + l < n1 { e[l] = row[c + l]; } }
            d[q] = load4(e, swap);
        }
    }
}
simd fn load_a(x: complex[:], swap: bool, n1: i64, n2: i64, c0: i64, w: i64, a: cv4[:]) { load_a_in(x, swap, n1, n2, c0, w, a); }

// Block column q (j1 = 4*(c0+q) + lane) at point k2 is at q + w*k2. Times
// w^(j1 k2), four k2 at a time are transposed into four rows of zt.
inline fn twiddle_transpose_in(p, c0: i64, w: i64, a: cv4[:], z: cv4[:]) {
    let n1 = p.n1;
    let n2 = p.n2;
    let g2 = p.g2;
    let g1 = p.g1;
    assert(a.len == w * n2 && z.len == g2 * n1);
    let mt = p.data[p.mid_off..p.mid_off + g1 * n2];
    for q in w {
        let g = c0 + q;
        let tw = mt[g * n2..g * n2 + n2];
        assert(tw.len == n2);
        if 4 * g + 4 <= n1 && n2 % 4 == 0 {
            // Four whole rows of zt, four whole k2 at a time: a 4x4 transpose.
            let z0 = g2 * 4 * g;
            let r0 = z[z0..z0 + g2];
            let r1 = z[z0 + g2..z0 + 2 * g2];
            let r2 = z[z0 + 2 * g2..z0 + 3 * g2];
            let r3 = z[z0 + 3 * g2..z0 + 4 * g2];
            assert(r0.len == g2 && r1.len == g2 && r2.len == g2 && r3.len == g2);
            for gk in g2 {
                let k = 4 * gk;
                let u0 = twid(a[q + w * k], tw[k]);
                let u1 = twid(a[q + w * (k + 1)], tw[k + 1]);
                let u2 = twid(a[q + w * (k + 2)], tw[k + 2]);
                let u3 = twid(a[q + w * (k + 3)], tw[k + 3]);
                r0[gk] = cv4 { v4 { u0.re.a, u1.re.a, u2.re.a, u3.re.a }, v4 { u0.im.a, u1.im.a, u2.im.a, u3.im.a } };
                r1[gk] = cv4 { v4 { u0.re.b, u1.re.b, u2.re.b, u3.re.b }, v4 { u0.im.b, u1.im.b, u2.im.b, u3.im.b } };
                r2[gk] = cv4 { v4 { u0.re.c, u1.re.c, u2.re.c, u3.re.c }, v4 { u0.im.c, u1.im.c, u2.im.c, u3.im.c } };
                r3[gk] = cv4 { v4 { u0.re.d, u1.re.d, u2.re.d, u3.re.d }, v4 { u0.im.d, u1.im.d, u2.im.d, u3.im.d } };
            }
            continue;
        }
        let rows = min(4, n1 - 4 * g);
        for gk in g2 {
            // Four consecutive k2 of this group's four columns, twiddled.
            var er: f64[16] = [0.0; 16];
            var ei: f64[16] = [0.0; 16];
            for u in 4 {
                let k2 = 4 * gk + u;
                if k2 < n2 {
                    let v = twid(a[q + w * k2], tw[k2]);
                    er[u] = v.re.a; er[4 + u] = v.re.b; er[8 + u] = v.re.c; er[12 + u] = v.re.d;
                    ei[u] = v.im.a; ei[4 + u] = v.im.b; ei[8 + u] = v.im.c; ei[12 + u] = v.im.d;
                }
            }
            // Row j1 = 4g + l of zt gets lane l of the four.
            for l in rows {
                let j1 = 4 * g + l;
                z[gk + g2 * j1] = cv4 { v4 { er[4 * l], er[4 * l + 1], er[4 * l + 2], er[4 * l + 3] },
                                        v4 { ei[4 * l], ei[4 * l + 1], ei[4 * l + 2], ei[4 * l + 3] } };
            }
        }
    }
}
simd fn twiddle_transpose(p, c0: i64, w: i64, a: cv4[:], z: cv4[:]) { twiddle_transpose_in(p, c0, w, a, z); }

inline fn load_b_in(z: cv4[:], g2: i64, n1: i64, d0: i64, w: i64, b: cv4[:]) {
    assert(b.len == w * n1);
    for j1 in n1 {
        let src = z[d0 + g2 * j1..d0 + g2 * j1 + w];
        let d = b[w * j1..w * j1 + w];
        assert(src.len == w && d.len == w);
        for q in w { d[q] = src[q]; }
    }
}
simd fn load_b(z: cv4[:], g2: i64, n1: i64, d0: i64, w: i64, b: cv4[:]) { load_b_in(z, g2, n1, d0, w, b); }

fn store4(e: complex[:], v: cv4) {
    e[0] = complex { v.re.a, v.im.a };
    e[1] = complex { v.re.b, v.im.b };
    e[2] = complex { v.re.c, v.im.c };
    e[3] = complex { v.re.d, v.im.d };
}

inline fn store_b_in(y: complex[:], swap: bool, scale: f64, n1: i64, n2: i64, d0: i64, w: i64, b: cv4[:]) {
    assert(b.len == w * n1 && y.len == n1 * n2);
    let sc = splat(scale);
    let full = max(0, min(w, n2 / 4 - d0));
    for k1 in n1 {
        let row = y[n2 * k1..n2 * k1 + n2];
        let src = b[w * k1..w * k1 + w];
        assert(row.len == n2 && src.len == w && 4 * (d0 + full) <= n2);
        for q in full {
            let e = row[4 * (d0 + q)..4 * (d0 + q) + 4];
            assert(e.len == 4);
            let v = src[q];
            let u = if swap { cv4 { v.im * sc, v.re * sc } } else { cv4 { v.re * sc, v.im * sc } };
            store4(e, u);
        }
        // The group past the last full one: only its columns below n2.
        for q in full..w {
            let c = 4 * (d0 + q);
            let v = src[q];
            let u = if swap { cv4 { v.im * sc, v.re * sc } } else { cv4 { v.re * sc, v.im * sc } };
            var e: complex[4] = [complex { 0.0, 0.0 }; 4];
            store4(e, u);
            for l in 4 { if c + l < n2 { row[c + l] = e[l]; } }
        }
    }
}
simd fn store_b(y: complex[:], swap: bool, scale: f64, n1: i64, n2: i64, d0: i64, w: i64, b: cv4[:]) { store_b_in(y, swap, scale, n1, n2, d0, w, b); }

// Real arrays read and written as complex pairs (x[2k], x[2k+1]): how a real
// transform of even length n packs its input into n/2 complex points.
fn load8(e: f64[:], swap: bool) -> cv4 {
    let re = v4 { e[0], e[2], e[4], e[6] };
    let im = v4 { e[1], e[3], e[5], e[7] };
    return if swap { cv4 { im, re } } else { cv4 { re, im } };
}

inline fn load_a_in(x: f64[:], swap: bool, n1: i64, n2: i64, c0: i64, w: i64, a: cv4[:]) {
    assert(a.len == w * n2 && x.len == 2 * n1 * n2);
    let full = max(0, min(w, n1 / 4 - c0));
    for j2 in n2 {
        let row = x[2 * n1 * j2..2 * n1 * j2 + 2 * n1];
        let d = a[w * j2..w * j2 + w];
        assert(row.len == 2 * n1 && d.len == w && 4 * (c0 + full) <= n1);
        for q in full {
            let e = row[8 * (c0 + q)..8 * (c0 + q) + 8];
            assert(e.len == 8);
            d[q] = load8(e, swap);
        }
        for q in full..w {
            let c = 4 * (c0 + q);
            var e: f64[8] = [0.0; 8];
            for l in 4 { if c + l < n1 { e[2 * l] = row[2 * (c + l)]; e[2 * l + 1] = row[2 * (c + l) + 1]; } }
            d[q] = load8(e, swap);
        }
    }
}
simd fn load_a(x: f64[:], swap: bool, n1: i64, n2: i64, c0: i64, w: i64, a: cv4[:]) { load_a_in(x, swap, n1, n2, c0, w, a); }

fn store8(e: f64[:], v: cv4) {
    e[0] = v.re.a; e[1] = v.im.a;
    e[2] = v.re.b; e[3] = v.im.b;
    e[4] = v.re.c; e[5] = v.im.c;
    e[6] = v.re.d; e[7] = v.im.d;
}

inline fn store_b_in(y: f64[:], swap: bool, scale: f64, n1: i64, n2: i64, d0: i64, w: i64, b: cv4[:]) {
    assert(b.len == w * n1 && y.len == 2 * n1 * n2);
    let sc = splat(scale);
    let full = max(0, min(w, n2 / 4 - d0));
    for k1 in n1 {
        let row = y[2 * n2 * k1..2 * n2 * k1 + 2 * n2];
        let src = b[w * k1..w * k1 + w];
        assert(row.len == 2 * n2 && src.len == w && 4 * (d0 + full) <= n2);
        for q in full {
            let e = row[8 * (d0 + q)..8 * (d0 + q) + 8];
            assert(e.len == 8);
            let v = src[q];
            store8(e, if swap { cv4 { v.im * sc, v.re * sc } } else { cv4 { v.re * sc, v.im * sc } });
        }
        for q in full..w {
            let c = 4 * (d0 + q);
            let v = src[q];
            var e: f64[8] = [0.0; 8];
            store8(e, if swap { cv4 { v.im * sc, v.re * sc } } else { cv4 { v.re * sc, v.im * sc } });
            for l in 4 { if c + l < n2 { row[2 * (c + l)] = e[2 * l]; row[2 * (c + l) + 1] = e[2 * l + 1]; } }
        }
    }
}
simd fn store_b(y: f64[:], swap: bool, scale: f64, n1: i64, n2: i64, d0: i64, w: i64, b: cv4[:]) { store_b_in(y, swap, scale, n1, n2, d0, w, b); }

// The number of complex points in an array the core reads or writes.
fn points(x: complex[:]) -> i64 { x.len }
fn points(x: f64[:]) -> i64 { x.len / 2 }

simd fn run_codelet(v: cv4[:]) { codelet(v); }

// The whole transform when n1 and n2 both have codelets (lengths up to 64):
// both steps in one function, so the moves and codelets inline into it
// rather than each being a call.
simd fn small_transform(p, x, y, inverse: bool, scale: f64, z: cv4[:], a: cv4[:]) {
    let n1 = p.n1;
    let n2 = p.n2;
    for g in p.g1 {
        let v = a[..n2];
        load_a_in(x, inverse, n1, n2, g, 1, v);
        codelet(v);
        twiddle_transpose_in(p, g, 1, v, z);
    }
    for gk in p.g2 {
        let v = a[..n1];
        load_b_in(z, p.g2, n1, gk, 1, v);
        codelet(v);
        store_b_in(y, inverse, scale, n1, n2, gk, 1, v);
    }
}

// The core transform of length p.len from x to y, each a complex[:] or a real
// array read as complex pairs: forward and unnormalized, or with `inverse`
// the inverse times n, each times `scale`. x and y may be the same array.
fn transform(p, x, y, inverse: bool, scale: f64) {
    let n1 = p.n1;
    let n2 = p.n2;
    let g1 = p.g1;
    let g2 = p.g2;
    assert(points(x) == p.len && points(y) == p.len);
    let zn = zt_size(p);
    let cap = block_cap(p);
    let bs = cap + PAD;
    let work = p.data[p.work_off..p.work_off + work_size(p)];
    let z = work[..zn];
    let a = work[zn + PAD..zn + PAD + cap];
    let t = work[zn + PAD + bs..zn + PAD + bs + cap];
    if p.codelets {
        // Each column group straight through its codelet: one sweep a step.
        small_transform(p, x, y, inverse, scale, z, a);
        return;
    }
    var c0 = 0;
    while c0 < g1 {
        let w = min(p.wa, g1 - c0);
        let size = w * n2;
        load_a(x, inverse, n1, n2, c0, w, a[..size]);
        if run(p.sa, p.data, w, a[..size], t[..size]) { twiddle_transpose(p, c0, w, a[..size], z); }
        else { twiddle_transpose(p, c0, w, t[..size], z); }
        c0 += w;
    }
    if p.wb == g2 {
        // One block holds all of zt: run the passes on it where it is.
        if run(p.sb, p.data, g2, z, t[..zn]) { store_b(y, inverse, scale, n1, n2, 0, g2, z); }
        else { store_b(y, inverse, scale, n1, n2, 0, g2, t[..zn]); }
        return;
    }
    var d0 = 0;
    while d0 < g2 {
        let w = min(p.wb, g2 - d0);
        let size = w * n1;
        load_b(z, g2, n1, d0, w, a[..size]);
        if run(p.sb, p.data, w, a[..size], t[..size]) { store_b(y, inverse, scale, n1, n2, d0, w, a[..size]); }
        else { store_b(y, inverse, scale, n1, n2, d0, w, t[..size]); }
        d0 += w;
    }
}

// Bluestein's algorithm, x to y, both of length p.cn: X[k] = c[k] * sum_j
// (x[j] c[j]) conj(c[k - j]), with the sum a circular convolution of length
// p.len done by two core transforms.
fn bluestein(p, x: complex[:], y: complex[:], inverse: bool, scale: f64) {
    let n = p.cn;
    let m = p.len;
    assert(x.len == n && y.len == n);
    var a: complex[>..] = [];
    a.resize(m, complex { 0.0, 0.0 });
    for k in n {
        let c = get_complex(p.data, p.chirp_off, k);
        let v = if inverse { complex { x[k].im, x[k].re } } else { x[k] };
        a[k] = complex { v.re * c.re - v.im * c.im, v.re * c.im + v.im * c.re };
    }
    transform(p, a, a, false, 1.0);
    // Times the filter's transform, then back: the inverse is the forward
    // transform of the swapped parts.
    for k in m {
        let f = get_complex(p.data, p.filt_off, k);
        let v = a[k];
        a[k] = complex { v.re * f.im + v.im * f.re, v.re * f.re - v.im * f.im };
    }
    transform(p, a, a, false, 1.0);
    for k in n {
        let c = get_complex(p.data, p.chirp_off, k);
        let v = complex { a[k].im, a[k].re };
        let re = (v.re * c.re - v.im * c.im) * scale;
        let im = (v.re * c.im + v.im * c.re) * scale;
        y[k] = if inverse { complex { im, re } } else { complex { re, im } };
    }
}

// ---------------------------------------------------------------------------
// Complex transforms

// The transform of x with plan p into y: forward unnormalized, or inverse
// divided by n. x and y may be the same array.
fn execute_into(p, x: complex[:], y: complex[:], inverse: bool) {
    assert(p.kind == KIND_COMPLEX && x.len == p.n && y.len == p.n);
    let scale = if inverse { 1.0 / p.n } else { 1.0 };
    if p.blue { bluestein(p, x, y, inverse, scale); }
    else { transform(p, x, y, inverse, scale); }
}

// The forward transform of x with plan p, unnormalized.
fn execute(p, x: complex[:]) -> complex[>..] {
    var y: complex[>..] = [];
    y.resize(p.n, complex { 0.0, 0.0 });
    execute_into(p, x, y, false);
    return y;
}

// The inverse transform of x with plan p, divided by n.
fn execute_inverse(p, x: complex[:]) -> complex[>..] {
    var y: complex[>..] = [];
    y.resize(p.n, complex { 0.0, 0.0 });
    execute_into(p, x, y, true);
    return y;
}

fn fft(x: complex[:]) -> complex[>..] {
    let p = plan_complex(x.len);
    return execute(p, x);
}

fn ifft(x: complex[:]) -> complex[>..] {
    let p = plan_complex(x.len);
    return execute_inverse(p, x);
}

// ---------------------------------------------------------------------------
// Real transforms
//
// A real input of even length n is n/2 complex points z[j] = x[2j] + i
// x[2j+1]. Their transform Z gives the spectrum of the even and odd samples,
// E[k] = (Z[k] + conj Z[h-k]) / 2 and O[k] = (Z[k] - conj Z[h-k]) / 2i, and
// X[k] = E[k] + w^k O[k], w = exp(-2 pi i / n): half a transform and one pass
// over the result. Odd lengths go through the complex transform.

// A plan for real transforms of length n: rfft and irfft.
fn plan_real(n: i64) -> plan {
    assert(n >= 1);
    var p = new_plan(n, KIND_REAL);
    if n % 2 == 1 {
        setup_complex(p, n);
        return p;
    }
    let h = n / 2;
    setup_complex(p, h);
    var tw: complex[>..] = [];
    for k in h / 2 + 1 { let c, s = root(k, n); tw.push(complex { c, s }); }
    p.real_off = p.data.len;
    put_complex(p.data, tw);
    p.realv_off = p.data.len;
    put_complex(p.data, tw[1..]);
    return p;
}

fn reversed(v: cv4) -> cv4 { cv4 { v4 { v.re.d, v.re.c, v.re.b, v.re.a }, v4 { v.im.d, v.im.c, v.im.b, v.im.a } } }

// The real transform's last pass, four k at a time from k = 1, each paired
// with h - k, while the two groups of four stay apart; returns the first k
// left for the scalar loop.
simd fn post_real(y: complex[:], h: i64, tw: cv4[:]) -> i64 {
    let half = splat(0.5);
    var k = 1;
    var i = 0;
    while 2 * k + 6 < h && i < tw.len {
        let fa = y[k..k + 4];
        let fb = y[h - k - 3..h - k + 1];
        assert(fa.len == 4 && fb.len == 4);
        let a = load4(fa, false);
        let b = reversed(load4(fb, false));
        let w = tw[i];
        let er = half * (a.re + b.re);
        let ei = half * (a.im - b.im);
        let or = half * (a.im + b.im);
        let oi = half * (b.re - a.re);
        let tr = w.re * or - w.im * oi;
        let ti = w.re * oi + w.im * or;
        store4(fa, cv4 { er + tr, ei + ti });
        store4(fb, reversed(cv4 { er - tr, ti - ei }));
        k += 4;
        i++;
    }
    return k;
}

// The inverse's first pass, x to z, the same way; returns the first k left.
simd fn pre_real(x: complex[:], z: complex[:], h: i64, tw: cv4[:]) -> i64 {
    let half = splat(0.5);
    var k = 1;
    var i = 0;
    while 2 * k + 6 < h && i < tw.len {
        let fa = x[k..k + 4];
        let fb = x[h - k - 3..h - k + 1];
        let za = z[k..k + 4];
        let zb = z[h - k - 3..h - k + 1];
        assert(fa.len == 4 && fb.len == 4 && za.len == 4 && zb.len == 4);
        let a = load4(fa, false);
        let b = reversed(load4(fb, false));
        let w = tw[i];
        let er = half * (a.re + b.re);
        let ei = half * (a.im - b.im);
        let dr = half * (a.re - b.re);
        let di = half * (a.im + b.im);
        let or = dr * w.re + di * w.im;
        let oi = di * w.re - dr * w.im;
        store4(za, cv4 { er - oi, ei + or });
        store4(zb, reversed(cv4 { er + oi, or - ei }));
        k += 4;
        i++;
    }
    return k;
}

// The complex core of p (length p.cn) from x to y.
fn core(p, x, y, inverse: bool, scale: f64) {
    if !p.blue {
        transform(p, x, y, inverse, scale);
        return;
    }
    // Bluestein runs on complex arrays: real pairs go through one.
    var a: complex[>..] = [];
    a.resize(p.cn, complex { 0.0, 0.0 });
    copy_points(x, a);
    bluestein(p, a, a, inverse, scale);
    copy_points(a, y);
}

fn copy_points(x: complex[:], y: complex[:]) { for v, k in x { y[k] = v; } }
fn copy_points(x: f64[:], y: complex[:]) { for k in y.len { y[k] = complex { x[2 * k], x[2 * k + 1] }; } }
fn copy_points(x: complex[:], y: f64[:]) { for v, k in x { y[2 * k] = v.re; y[2 * k + 1] = v.im; } }

// The spectrum of the real x with plan p into y, n/2 + 1 values (the rest
// are their conjugates), unnormalized.
fn execute_real_into(p, x: f64[:], y: complex[:]) {
    let n = p.n;
    assert(p.kind == KIND_REAL && x.len == n && y.len == n / 2 + 1);
    if n % 2 == 1 {
        var c: complex[>..] = [];
        for v in x { c.push(complex { v, 0.0 }); }
        var z: complex[>..] = [];
        z.resize(n, complex { 0.0, 0.0 });
        core(p, c, z, false, 1.0);
        for k in y.len { y[k] = z[k]; }
        return;
    }
    let h = n / 2;
    core(p, x, y[..h], false, 1.0);
    let z0 = y[0];
    y[0] = complex { z0.re + z0.im, 0.0 };
    y[h] = complex { z0.re - z0.im, 0.0 };
    let tv = p.data[p.realv_off..p.realv_off + (h / 2 + 3) / 4];
    let k0 = post_real(y, h, tv);
    for k in k0..h / 2 + 1 {
        let a = y[k];
        let b = y[h - k];
        // E = (a + conj b) / 2, O = (a - conj b) / 2i, X[k] = E + w^k O and
        // X[h-k] = conj(E - w^k O).
        let er = 0.5 * (a.re + b.re);
        let ei = 0.5 * (a.im - b.im);
        let or = 0.5 * (a.im + b.im);
        let oi = 0.5 * (b.re - a.re);
        let w = get_complex(p.data, p.real_off, k);
        let tr = w.re * or - w.im * oi;
        let ti = w.re * oi + w.im * or;
        y[k] = complex { er + tr, ei + ti };
        y[h - k] = complex { er - tr, ti - ei };
    }
}

// The real signal of length n whose spectrum's first n/2 + 1 values are x,
// into y: the inverse of execute_real_into, divided by n. x[0] and, for even
// n, x[n/2] should be real; their imaginary parts are ignored.
fn execute_real_inverse_into(p, x: complex[:], y: f64[:]) {
    let n = p.n;
    assert(p.kind == KIND_REAL && x.len == n / 2 + 1 && y.len == n);
    if n % 2 == 1 {
        var c: complex[>..] = [];
        c.resize(n, complex { 0.0, 0.0 });
        c[0] = complex { x[0].re, 0.0 };
        for k in 1..x.len {
            c[k] = x[k];
            c[n - k] = complex { x[k].re, 0.0 - x[k].im };
        }
        core(p, c, c, true, 1.0 / n);
        for k in n { y[k] = c[k].re; }
        return;
    }
    let h = n / 2;
    var z: complex[>..] = [];
    z.resize(h, complex { 0.0, 0.0 });
    // E[k] = (X[k] + conj X[h-k]) / 2, O[k] = (X[k] - conj X[h-k]) w^-k / 2,
    // Z[k] = E[k] + i O[k], and Z[h-k] = conj E[k] + i conj O[k].
    let tv = p.data[p.realv_off..p.realv_off + (h / 2 + 3) / 4];
    let k0 = pre_real(x, z, h, tv);
    for k in h / 2 + 1 {
        if k > 0 && k < k0 { continue; }
        let a = if k == 0 { complex { x[0].re, 0.0 } } else { x[k] };
        let b = if k == 0 { complex { x[h].re, 0.0 } } else { x[h - k] };
        let er = 0.5 * (a.re + b.re);
        let ei = 0.5 * (a.im - b.im);
        let dr = 0.5 * (a.re - b.re);
        let di = 0.5 * (a.im + b.im);
        let w = get_complex(p.data, p.real_off, k);
        let or = dr * w.re + di * w.im;            // times conj w
        let oi = di * w.re - dr * w.im;
        z[k] = complex { er - oi, ei + or };
        if k > 0 && h - k != k { z[h - k] = complex { er + oi, or - ei }; }
    }
    core(p, z, y, true, 1.0 / h);
}

fn execute_real(p, x: f64[:]) -> complex[>..] {
    var y: complex[>..] = [];
    y.resize(p.n / 2 + 1, complex { 0.0, 0.0 });
    execute_real_into(p, x, y);
    return y;
}

fn execute_real_inverse(p, x: complex[:]) -> f64[>..] {
    var y: f64[>..] = [];
    y.resize(p.n, 0.0);
    execute_real_inverse_into(p, x, y);
    return y;
}

// The spectrum of real x: its first x.len / 2 + 1 values.
fn rfft(x: f64[:]) -> complex[>..] {
    let p = plan_real(x.len);
    return execute_real(p, x);
}

// The real signal of length n with spectrum x (n / 2 + 1 values).
fn irfft(x: complex[:], n: i64) -> f64[>..] {
    let p = plan_real(n);
    return execute_real_inverse(p, x);
}

// ---------------------------------------------------------------------------
// Two dimensions

// The 2-D transform of x, rows by cols, row-major, in place: every row, then
// every column. Forward unnormalized, or inverse divided by rows * cols.
fn transform2_into(x: complex[:], rows: i64, cols: i64, inverse: bool) {
    assert(rows >= 1 && cols >= 1 && x.len == rows * cols);
    let pr = plan_complex(cols);
    for r in rows { execute_into(pr, x[r * cols..(r + 1) * cols], x[r * cols..(r + 1) * cols], inverse); }
    let pc = plan_complex(rows);
    var col: complex[>..] = [];
    col.resize(rows, complex { 0.0, 0.0 });
    for c in cols {
        for r in rows { col[r] = x[r * cols + c]; }
        execute_into(pc, col, col, inverse);
        for r in rows { x[r * cols + c] = col[r]; }
    }
}

fn fft2(x: complex[:], rows: i64, cols: i64) -> complex[>..] {
    var y: complex[>..] = [];
    y.append(x);
    transform2_into(y, rows, cols, false);
    return y;
}

fn ifft2(x: complex[:], rows: i64, cols: i64) -> complex[>..] {
    var y: complex[>..] = [];
    y.append(x);
    transform2_into(y, rows, cols, true);
    return y;
}

// ---------------------------------------------------------------------------
// Helpers

// The frequency of each bin of a length-n transform, for samples d apart:
// 0, 1, ..., then the negative ones, over n d. As NumPy's fftfreq.
fn fftfreq(n: i64, d: f64 = 1.0) -> f64[>..] {
    var out: f64[>..] = [];
    for k in n { out.push((if k < (n + 1) / 2 { k } else { k - n }) / (n * d)); }
    return out;
}

// The frequencies of rfft's n / 2 + 1 bins.
fn rfftfreq(n: i64, d: f64 = 1.0) -> f64[>..] {
    var out: f64[>..] = [];
    for k in n / 2 + 1 { out.push(k / (n * d)); }
    return out;
}

// x rotated so the zero frequency is in the middle, as NumPy's fftshift.
fn fftshift<T>(x: T[:]) -> T[>..] {
    var out: T[>..] = [];
    let n = x.len;
    let s = n / 2;
    for k in n { out.push(x[(k + n - s) % n]); }
    return out;
}

// fftshift undone.
fn ifftshift<T>(x: T[:]) -> T[>..] {
    var out: T[>..] = [];
    let n = x.len;
    let s = n / 2;
    for k in n { out.push(x[(k + s) % n]); }
    return out;
}

// The linear convolution of real a and b, a.len + b.len - 1 values, by
// real transforms padded to a fast length.
fn convolve(a: f64[:], b: f64[:]) -> f64[>..] {
    assert(a.len >= 1 && b.len >= 1);
    let m = a.len + b.len - 1;
    let n = smooth_at_least(m);
    let p = plan_real(n);
    var pa: f64[>..] = [];
    pa.resize(n, 0.0);
    for v, i in a { pa[i] = v; }
    var pb: f64[>..] = [];
    pb.resize(n, 0.0);
    for v, i in b { pb[i] = v; }
    var fa = execute_real(p, pa);
    let fb = execute_real(p, pb);
    for &v, k in fa { let u = fb[k]; v = complex { v.re * u.re - v.im * u.im, v.re * u.im + v.im * u.re }; }
    let full = execute_real_inverse(p, fa);
    var out: f64[>..] = [];
    out.append(full[..m]);
    return out;
}
