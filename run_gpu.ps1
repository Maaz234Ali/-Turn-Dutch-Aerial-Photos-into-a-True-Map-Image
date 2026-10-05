# run_gpu.ps1
# GPU-enabled launcher for the True Orthophoto pipeline.
# Sets all required CUDA environment variables for the current shell session,
# then forwards all arguments to main.py.
# Usage:  .\run_gpu.ps1 [main.py arguments...]
# Example: .\run_gpu.ps1 --workspace . --bbox 82798 445352 86084 446362 --gsd 0.25 --out-dir output

$CUDA_ROOT = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'

$env:CUDA_PATH             = $CUDA_ROOT
$env:CUDA_HOME             = $CUDA_ROOT
$env:NUMBAPRO_NVVM         = "$CUDA_ROOT\nvvm\bin\nvvm64_40_0.dll"
$env:NUMBAPRO_LIBDEVICE    = "$CUDA_ROOT\nvvm\libdevice"
$env:PATH                  = "$CUDA_ROOT\bin;$CUDA_ROOT\nvvm\bin;" + $env:PATH

Write-Host "[GPU Launcher] CUDA_HOME   = $env:CUDA_HOME" -ForegroundColor Cyan
Write-Host "[GPU Launcher] NVVM DLL    = $env:NUMBAPRO_NVVM" -ForegroundColor Cyan

# Quick CUDA sanity check
$check = python -c "from numba import cuda; print('CUDA:', cuda.is_available())" 2>&1
Write-Host "[GPU Launcher] $check" -ForegroundColor $(if ($check -like '*True*') {'Green'} else {'Red'})

# Run the pipeline
python -u main.py @args
