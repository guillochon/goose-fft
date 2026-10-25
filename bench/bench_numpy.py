"""The same transforms as bench.goose, through NumPy and SciPy (both pocketfft).

    python bench/bench_numpy.py [label]      -> bench_<label>_numpy.csv and bench_<label>_scipy.csv

NumPy's np.fft and SciPy's scipy.fft are both pocketfft (C++, with SIMD in
SciPy's build), each run single-threaded here (scipy.fft's workers=1). Every
call pays Python's overhead, a microsecond or two, which dominates the
smallest lengths. The same timing rule as the Goose side: the best time per
call over samples of about a millisecond, at least 5 samples and 0.2 s.
"""
import sys
import time
import numpy as np
import scipy.fft

POW2 = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
MIXED = [100, 1000, 3000, 10000, 100000, 1000000]
PRIME = [1009, 10007]
label = sys.argv[1] if len(sys.argv) > 1 else "ref"


def time_it(rows, kind, n, f):
    reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            f()
        if time.perf_counter() - t0 > 0.001 or reps > 1000000:
            break
        reps *= 2
    best, total, samples = float("inf"), 0.0, 0
    while samples < 5 or total < 0.2:
        t0 = time.perf_counter()
        for _ in range(reps):
            f()
        dt = time.perf_counter() - t0
        best, total, samples = min(best, dt / reps), total + dt, samples + 1
    rows.append((kind, n, best))
    flops = 5 * n * np.log2(n) / (2 if kind == "rfft" else 1)
    print(f"{kind} n={n}: {best * 1e6:.3f} us  {flops / best / 1e6:.0f} MFLOPS")


rng = np.random.default_rng(42)
for name, cfft, rfft in [("numpy", np.fft.fft, np.fft.rfft),
                         ("scipy", lambda x: scipy.fft.fft(x, workers=1), lambda x: scipy.fft.rfft(x, workers=1))]:
    print(f"--- {name}")
    rows = []
    for n in POW2 + MIXED + PRIME:
        x = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)
        cfft(x)                                     # plan cache warm
        time_it(rows, "fft", n, lambda: cfft(x))
    for n in POW2 + MIXED:
        x = rng.uniform(-1, 1, n)
        rfft(x)
        time_it(rows, "rfft", n, lambda: rfft(x))
    with open(f"bench_{label}_{name}.csv", "w") as f:
        f.write("kind,n,seconds\n")
        for kind, n, secs in rows:
            f.write(f"{kind},{n},{secs}\n")
