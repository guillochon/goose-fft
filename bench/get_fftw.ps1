# Fetches FFTW 3.3.10, builds it with clang-cl and AVX2 (and FMA, as FFTW's
# own --enable-avx2 does), and builds bench_fftw.exe against it: the
# reference the benchmark compares with. Everything goes in
# bench\third_party, which git ignores.
#
#   powershell -File bench\get_fftw.ps1
#
# Needs Visual Studio 2026 (18) with its clang-cl and CMake; -VsDir points
# elsewhere. FFTW's CMakeLists passes /bigobj as a preprocessor definition,
# which clang-cl rejects, so the unpacked copy gets a one-line patch.
param([string]$VsDir = "C:\Program Files\Microsoft Visual Studio\18\Community")

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$tp = Join-Path $here "third_party"
New-Item -ItemType Directory -Force $tp | Out-Null
$tar = Join-Path $tp "fftw-3.3.10.tar.gz"
if (-not (Test-Path $tar)) {
    Invoke-WebRequest -Uri "https://www.fftw.org/fftw-3.3.10.tar.gz" -OutFile $tar
}
$md5 = (Get-FileHash -Algorithm MD5 $tar).Hash.ToLower()
if ($md5 -ne "8ccbf6a5ea78a16dbc3e1306e234cc5c") { throw "fftw-3.3.10.tar.gz: unexpected MD5 $md5" }
$src = Join-Path $tp "fftw-3.3.10"
if (-not (Test-Path $src)) { tar -xzf $tar -C $tp }
$cm = Join-Path $src "CMakeLists.txt"
(Get-Content $cm) -replace 'target_compile_definitions \(\$\{fftw3_lib\} PRIVATE /bigobj\)', 'target_compile_options (${fftw3_lib} PRIVATE /bigobj)' |
    Set-Content -Encoding ascii $cm

$bat = Join-Path $tp "build.bat"
@"
@echo off
call "$VsDir\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "$tp"
cmake -S fftw-3.3.10 -B fftw-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_C_FLAGS="/O2 /arch:AVX2" ^
  -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DENABLE_SSE2=ON -DENABLE_AVX=ON -DENABLE_AVX2=ON ^
  -DCMAKE_INSTALL_PREFIX="$tp\fftw-install" || exit /b 1
cmake --build fftw-build --config Release || exit /b 1
cmake --install fftw-build || exit /b 1
cd /d "$here"
clang-cl /nologo /MD /O2 /arch:AVX2 /I "$tp\fftw-install\include" bench_fftw.c "$tp\fftw-install\lib\fftw3.lib" /Fe:bench_fftw.exe || exit /b 1
"@ | Set-Content -Encoding ascii $bat
cmd /c $bat
if ($LASTEXITCODE -ne 0) { throw "building FFTW or the harness failed" }
Write-Output "built $here\bench_fftw.exe"
