$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = Join-Path $root "venv311\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Project Python environment not found: $python"
}

$env:ISL_ALLOWED_ORIGINS = "http://127.0.0.1:5173"

Start-Process -FilePath $python -ArgumentList "-m uvicorn app.api:app --host 127.0.0.1 --port 8000" -WorkingDirectory $root
Start-Process -FilePath $python -ArgumentList "-m http.server 5173 --directory frontend" -WorkingDirectory $root

Write-Host "ISL Recognition is starting."
Write-Host "Web app: http://127.0.0.1:5173"
Write-Host "API:     http://127.0.0.1:8000/health"
Write-Host "Camera access requires localhost or HTTPS in the browser."