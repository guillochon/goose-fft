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


def mflops(kind, n, t):
    return 5 * n * math.log2(n) / (2 if kind == "rfft" else 1) / t / 1e6


cols = [(f"goose {label}", data) for label, data in goose.items() if data] + refs
keys = [k for k in now if fftw and k in fftw] if fftw else list(now)
print(f"{'kind':5}{'n':>9} " + " ".join(f"{name[:12]:>13}" for name, _ in cols) + f" {'vs FFTW':>8} {'GF/s':>6}")
for k in keys:
    kind, n = k
    line = f"{kind:5}{n:9} " + " ".join(f"{data[k] * 1e6:11.3f}us" if k in data else f"{'-':>13}" for _, data in cols)
    if fftw and k in fftw:
        line += f" {now[k] / fftw[k]:7.2f}x"
    line += f" {mflops(kind, n, now[k]) / 1e3:6.1f}"
    print(line)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for kind, marker in [("fft", "o"), ("rfft", "s")]:
    ns = sorted(n for kd, n in now if kd == kind and n & (n - 1) == 0)
    if not ns:
        continue
    for i, (name, data) in enumerate([(f"Goose fft", now)] + refs):
        pts = [(n, mflops(kind, n, data[(kind, n)]) / 1e3) for n in ns if (kind, n) in data]
        if pts:
            ax[0 if kind == "fft" else 1].plot(*zip(*pts), marker + "-", label=name,
                                                lw=2.5 if i == 0 else 1.2)
for a, kind in zip(ax, ["complex", "real"]):
    a.set(xscale="log", xlabel="n (power of two)", ylabel="GFLOP/s (5 n log2 n / t" + (", halved)" if kind == "real" else ")"),
          title=f"{kind} forward transform, 1 thread")
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"$2^{{{int(round(math.log2(v)))}}}$"))
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
fig.suptitle("Goose fft vs FFTW 3.3.10 and pocketfft (NumPy/SciPy), clang-cl 22 /O2 /arch:AVX2, i5-12600K P-core")
fig.tight_layout()
fig.savefig("compare.png", dpi=110)
print("wrote compare.png")
