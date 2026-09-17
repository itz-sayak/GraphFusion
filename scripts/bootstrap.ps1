# One-shot local setup on Windows (PowerShell):  .\scripts\bootstrap.ps1 [-WithNycData]
param([switch]$WithNycData)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe scripts\make_sample_data.py
if ($WithNycData) { .\.venv\Scripts\python.exe scripts\download_demo_data.py }
Push-Location frontend; npm install; Pop-Location
Write-Host "`nReady. Backend: .\.venv\Scripts\uvicorn backend.main:app --port 8000   UI: cd frontend; npm run dev"
