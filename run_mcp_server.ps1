$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run: py -3.12 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

Write-Host "Starting SupportIQ MCP at http://127.0.0.1:8765/mcp" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" ".\mcp_server.py"
