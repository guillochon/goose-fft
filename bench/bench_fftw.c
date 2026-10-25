/* The same transforms as bench.goose, through FFTW 3.

     bench_fftw [out.csv] [measure|patient|estimate]

   Built by run_bench.ps1 with the same clang-cl and flags as the Goose
   benchmark, against an FFTW built with --enable-avx2 (CMake ENABLE_AVX2).
   Plans are out of place and made once per length, as a Goose plan is; the
   planning time is not counted. The same timing rule as the Goose side: the
   best time per call over samples of about a millisecond, at least 5 samples
   and 0.2 s. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <fftw3.h>

static double now(void) {
    static LARGE_INTEGER f;
    LARGE_INTEGER t;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}

static FILE *csv;
static double sink;
static unsigned flags = FFTW_MEASURE;

static double flops(const char *kind, long n) {
    double f = 5.0 * n * log2((double)n);
    return strcmp(kind, "rfft") == 0 ? f / 2 : f;
}

static void time_plan(const char *kind, long n, fftw_plan p, double *out0) {
    long reps = 1;
    for (;;) {
        double t0 = now();
        for (long i = 0; i < reps; i++) fftw_execute(p);
        if (now() - t0 > 0.001 || reps > 1000000) break;
        reps *= 2;
    }
    double best = 1e30, total = 0;
    int samples = 0;
    while (samples < 5 || total < 0.2) {
        double t0 = now();
        for (long i = 0; i < reps; i++) fftw_execute(p);
        double dt = now() - t0;
        if (dt / reps < best) best = dt / reps;
        total += dt;
        samples++;
    }
    sink += *out0;
    fprintf(csv, "%s,%ld,%.10g\n", kind, n, best);
    printf("%s n=%ld: %.3f us  %.0f MFLOPS\n", kind, n, best * 1e6, flops(kind, n) / best / 1e6);
}

static void bench_complex(long n) {
    fftw_complex *in = fftw_alloc_complex(n), *out = fftw_alloc_complex(n);
    fftw_plan p = fftw_plan_dft_1d((int)n, in, out, FFTW_FORWARD, flags);
    for (long i = 0; i < n; i++) { in[i][0] = rand() / (double)RAND_MAX - 0.5; in[i][1] = rand() / (double)RAND_MAX - 0.5; }
    time_plan("fft", n, p, &out[0][0]);
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

static void bench_real(long n) {
    double *in = fftw_alloc_real(n);
    fftw_complex *out = fftw_alloc_complex(n / 2 + 1);
    fftw_plan p = fftw_plan_dft_r2c_1d((int)n, in, out, flags);
    for (long i = 0; i < n; i++) in[i] = rand() / (double)RAND_MAX - 0.5;
    time_plan("rfft", n, p, &out[0][0]);
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

int main(int argc, char **argv) {
    static const long pow2[] = {16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};
    static const long mixed[] = {100, 1000, 3000, 10000, 100000, 1000000};
    static const long prime[] = {1009, 10007};
    const char *path = argc > 1 ? argv[1] : "bench_fftw.csv";
    if (argc > 2 && strcmp(argv[2], "patient") == 0) flags = FFTW_PATIENT;
    if (argc > 2 && strcmp(argv[2], "estimate") == 0) flags = FFTW_ESTIMATE;
    csv = fopen(path, "w");
    if (!csv) { fprintf(stderr, "cannot write %s\n", path); return 1; }
    fprintf(csv, "kind,n,seconds\n");
    for (int i = 0; i < 9; i++) bench_complex(pow2[i]);
    for (int i = 0; i < 6; i++) bench_complex(mixed[i]);
    for (int i = 0; i < 2; i++) bench_complex(prime[i]);
    for (int i = 0; i < 9; i++) bench_real(pow2[i]);
    for (int i = 0; i < 6; i++) bench_real(mixed[i]);
    fclose(csv);
    printf("checksum %g\n", sink);
    return 0;
}
