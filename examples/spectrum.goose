// Three everyday uses of fft: the spectrum of a noisy signal, smoothing by
// convolution, and multiplying big numbers through their digits' spectra.
//
//   goose examples/spectrum.goose

import std;
import math;
import fft from "..";

fn main() {
    // Two tones, 50 Hz and 120 Hz, in noise, sampled at 1 kHz for one second.
    let n = 1000;
    let rate = 1000.0;
    var r = rng { 7 };
    var signal: f64[>..] = [];
    for i in n {
        let t = i / rate;
        signal.push(sin(TAU * 50.0 * t) + 0.5 * sin(TAU * 120.0 * t) + 0.8 * (r.rand_flt() - 0.5));
    }
    let spectrum = fft::rfft(signal);
    let freqs = fft::rfftfreq(n, 1.0 / rate);
    // The two strongest bins: amplitude is |X| * 2 / n.
    var best: i64[2] = [0, 0];
    for v, k in spectrum {
        let a = sqrt(v.re * v.re + v.im * v.im);
        let b0 = spectrum[best[0]];
        if a > sqrt(b0.re * b0.re + b0.im * b0.im) { best[1] = best[0]; best[0] = k; continue; }
        let b1 = spectrum[best[1]];
        if a > sqrt(b1.re * b1.re + b1.im * b1.im) { best[1] = k; }
    }
    for k in best {
        let v = spectrum[k];
        print("peak at ", freqs[k], " Hz, amplitude ", 2.0 * sqrt(v.re * v.re + v.im * v.im) / n);
    }

    // A moving average of 25 samples, by convolution: the 120 Hz tone, whose
    // period is about 8 samples, mostly averages away; 50 Hz survives.
    var box: f64[>..] = [];
    for i in 25 { box.push(1.0 / 25); }
    let smooth = fft::convolve(signal, box);
    let s2 = fft::rfft(smooth[12..12 + n]);
    let a50 = s2[50];
    let a120 = s2[120];
    print("after smoothing: 50 Hz amplitude ", 2.0 * sqrt(a50.re * a50.re + a50.im * a50.im) / n,
          ", 120 Hz amplitude ", 2.0 * sqrt(a120.re * a120.re + a120.im * a120.im) / n);

    // 123456789 * 987654321 through convolution of decimal digits (least
    // significant first), then carrying.
    let a: f64[] = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let b: f64[] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let c = fft::convolve(a, b);
    var digits: i64[>..] = [];
    var carry = 0;
    for v in c {
        let d = (round(v) as i64) + carry;
        digits.push(d % 10);
        carry = d / 10;
    }
    while carry > 0 { digits.push(carry % 10); carry /= 10; }
    var text: u8[>..] = [];
    for d in digits.len { format(text, digits[digits.len - 1 - d]); }
    print("123456789 * 987654321 = ", text, "  (", 123456789 * 987654321, ")");
}
