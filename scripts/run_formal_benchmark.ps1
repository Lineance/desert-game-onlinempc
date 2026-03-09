param(
    [ValidateSet("3", "4", "both")]
    [string]$Level = "3",
    [int]$Episodes = 30,
    [int]$Seed = 42,
    [int]$Workers = 13,
    [int]$SolverThreads = 1,
    [switch]$CleanRaw
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir
$python = "uv run"

$rawDir = Join-Path $root "outputs/benchmarks/raw"

if ($Level -eq "both") {
    $summaryFile = Join-Path $root "outputs/benchmarks/summary/benchmark_summary_both.csv"
    $figDir = Join-Path $root "outputs/benchmarks/figures/both"
    $inputGlob = Join-Path $rawDir "*_benchmark_*.csv"
} elseif ($Level -eq "3") {
    $summaryFile = Join-Path $root "outputs/benchmarks/summary/benchmark_summary_level3.csv"
    $figDir = Join-Path $root "outputs/benchmarks/figures/level3"
    $inputGlob = Join-Path $rawDir "level3_benchmark_*.csv"
} else {
    $summaryFile = Join-Path $root "outputs/benchmarks/summary/benchmark_summary_level4.csv"
    $figDir = Join-Path $root "outputs/benchmarks/figures/level4"
    $inputGlob = Join-Path $rawDir "level4_benchmark_*.csv"
}

if ($CleanRaw) {
    if (Test-Path $rawDir) {
        if ($Level -eq "both") {
            Remove-Item -Path (Join-Path $rawDir "*_benchmark_*.csv") -Force -ErrorAction SilentlyContinue
            Write-Host "[clean] removed all raw benchmark files in $rawDir"
        } elseif ($Level -eq "3") {
            Remove-Item -Path (Join-Path $rawDir "level3_benchmark_*.csv") -Force -ErrorAction SilentlyContinue
            Write-Host "[clean] removed level3 raw benchmark files in $rawDir"
        } else {
            Remove-Item -Path (Join-Path $rawDir "level4_benchmark_*.csv") -Force -ErrorAction SilentlyContinue
            Write-Host "[clean] removed level4 raw benchmark files in $rawDir"
        }
    }
}

if ($Level -eq "3" -or $Level -eq "both") {
    Write-Host "[run] Level3 formal grid"
    & $python (Join-Path $scriptDir "benchmark_horizon_scenarios.py") `
        --level 3 `
        --episodes $Episodes `
        --seed $Seed `
        --workers $Workers `
        --solver-threads $SolverThreads `
        --horizons 2 4 6 8 10 `
        --scenarios 1 2 4 8
}

if ($Level -eq "4" -or $Level -eq "both") {
    Write-Host "[run] Level4 formal grid"
    & $python (Join-Path $scriptDir "benchmark_horizon_scenarios.py") `
        --level 4 `
        --episodes $Episodes `
        --seed ($Seed + 1000) `
        --workers $Workers `
        --solver-threads $SolverThreads `
        --horizons 4 8 12 16 20 30 `
        --scenarios 1 2 4 8 16
}

Write-Host "[summary] aggregate selected raw files"
& $python (Join-Path $scriptDir "summarize_benchmark.py") --input-glob $inputGlob --output $summaryFile

Write-Host "[plot] generate figures"
& $python (Join-Path $scriptDir "plot_benchmark.py") --summary $summaryFile --outdir $figDir

Write-Host "Benchmark completed."
Write-Host "- Workers: $Workers"
Write-Host "- SolverThreads: $SolverThreads"
Write-Host "- Raw:     $rawDir"
Write-Host "- Filter:  $inputGlob"
Write-Host "- Summary: $summaryFile"
Write-Host "- Figures: $figDir"
