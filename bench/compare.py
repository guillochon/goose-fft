"""Compare the Goose runs against FFTW, NumPy and SciPy: a table and compare.png.

    python compare.py [label ...]      (in this directory, after run_bench.ps1)

The headline comparison is against FFTW 3.3.10 with FFTW_MEASURE plans
(bench_fftw.csv), both built with clang-cl /O2 /arch:AVX2 and pinned to one
P-core. FFTW_PATIENT (bench_fftw_patient.csv) and NumPy/SciPy's pocketfft
(bench_ref_numpy.csv, bench_ref_scipy.csv, through Python) are shown when
their CSVs exist. Labels name Goose runs (bench_goose_<label>.csv); the
first is the one charted, "now" by default, and any others go in the table.
"""
import csv
import math
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {(r["kind"], int(r["n"])): float(r["seconds"]) for r in csv.DictReader(f)}


labels = sys.argv[1:] or ["now"]
goose = {label: load(f"bench_goose_{label}.csv") for label in labels}
now = goose[labels[0]]
fftw = load("bench_fftw.csv")
refs = [("FFTW measure", fftw), ("FFTW patient", load("bench_fftw_patient.csv")),
        ("NumPy", load("bench_ref_numpy.csv")), ("SciPy", load("bench_ref_scipy.csv"))]
refs = [(name, data) for name, data in refs if data]


def gflops(kind, n, t):
    return 5 * n * math.log2(n) / (2 if kind == "rfft" else 1) / t / 1e9


cols = [(f"goose {label}", data) for label, data in goose.items() if data] + refs
print(f"{'kind':5}{'n':>9} " + " ".join(f"{name[:12]:>13}" for name, _ in cols)
      + f" {'vs FFTW':>8} {'vs SciPy':>9} {'GF/s':>6}")
scipy = dict(refs).get("SciPy")
for k in now:
    kind, n = k
    line = f"{kind:5}{n:9} " + " ".join(f"{data[k] * 1e6:11.3f}us" if k in data else f"{'-':>13}" for _, data in cols)
    line += f" {now[k] / fftw[k]:7.2f}x" if fftw and k in fftw else f" {'-':>8}"
    line += f" {now[k] / scipy[k]:8.2f}x" if scipy and k in scipy else f" {'-':>9}"
    line += f" {gflops(kind, n, now[k]):6.1f}"
    print(line)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
styles = {"Goose fft": ("C3", "o-", 2.6), "FFTW measure": ("C0", "s-", 1.4), "FFTW patient": ("C0", "s:", 1.2),
          "NumPy": ("C2", "^-", 1.2), "SciPy": ("C1", "v-", 1.2)}
for a, kind, title in [(ax[0], "fft", "complex"), (ax[1], "rfft", "real")]:
    ns = sorted(n for kd, n in now if kd == kind and n & (n - 1) == 0)
    for name, data in [("Goose fft", now)] + refs:
        pts = [(n, gflops(kind, n, data[(kind, n)])) for n in ns if (kind, n) in data]
        if pts:
            color, style, lw = styles[name]
            a.plot(*zip(*pts), style, color=color, lw=lw, label=name)
    a.set(xscale="log", xlabel="n (power of two)",
          ylabel="GFLOP/s, " + ("2.5 n log2 n / t" if kind == "rfft" else "5 n log2 n / t"),
          title=f"{title} forward transform, one thread")
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"$2^{{{int(round(math.log2(v)))}}}$"))
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_ylim(bottom=0)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
fig.suptitle("Goose fft vs FFTW 3.3.10 and pocketfft (NumPy/SciPy); clang-cl 22 /O2 /arch:AVX2, i5-12600K P-core")
fig.tight_layout()
fig.savefig("compare.png", dpi=110)
print("wrote compare.png")
