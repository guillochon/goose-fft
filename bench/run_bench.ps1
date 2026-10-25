# Builds the Goose benchmark with the C compiler goose finds (clang-cl from
# Visual Studio 2026 on the test machine; GOOSE_CC picks another) and runs
# it, FFTW, and NumPy/SciPy, each pinned to one performance core so that a
# hybrid CPU (P- and E-cores) does not move them around mid-run.
#
#   powershell -File bench\run_bench.ps1 [-Core 4] [-Label now] [-Only goose|fftw|numpy]
#
# Writes bench_goose_<label>.csv, bench_fftw.csv (FFTW_MEASURE plans),
# bench_fftw_patient.csv (FFTW_PATIENT) and bench_ref_numpy.csv /
# bench_ref_scipy.csv. bench_fftw.exe is built by get_fftw.ps1, which
# fetches and builds FFTW into bench\third_party.
param([int]$Core = 4, [string]$Label = "now", [string]$Only = "", [string]$Flags = "/arch:AVX2")

$here = $PSScriptRoot
$goose = if ($env:GOOSE) { $env:GOOSE } else { "goose" }

function Run-Pinned($exe, $argList, $log, [long]$affinity) {
    $p = Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory $here -NoNewWindow -PassThru `
        -RedirectStandardOutput $log
    $p.ProcessorAffinity = [IntPtr]$affinity
    $p.WaitForExit()
}

$one = [long](1 -shl $Core)
if ($Only -eq "" -or $Only -eq "goose") {
    # Build failing loudly: a stale executable would quietly measure old code.
    $exe = "$here\bench_$Label.exe"
    Remove-Item $exe -ErrorAction SilentlyContinue
    if ($Flags) { $env:CL = $Flags } else { Remove-Item Env:CL -ErrorAction SilentlyContinue }
    & $goose build -o $exe "$here\bench.goose" | Out-Null
    Remove-Item Env:CL -ErrorAction SilentlyContinue
    if (-not (Test-Path $exe)) { throw "building bench.goose failed" }
    Run-Pinned $exe @("bench_goose_$Label.csv") "$here\log_goose_$Label.txt" $one
    Write-Output "goose $Label done"
}
if ($Only -eq "" -or $Only -eq "fftw") {
    Run-Pinned "$here\bench_fftw.exe" @("bench_fftw.csv", "measure") "$here\log_fftw.txt" $one
    Run-Pinned "$here\bench_fftw.exe" @("bench_fftw_patient.csv", "patient") "$here\log_fftw_patient.txt" $one
    Write-Output "fftw done"
}
if ($Only -eq "" -or $Only -eq "numpy") {
    Run-Pinned "python" @("bench_numpy.py", "ref") "$here\log_numpy.txt" $one
    Write-Output "numpy done"
}
