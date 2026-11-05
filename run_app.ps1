$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run: py -3.12 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

& ".\.venv\Scripts\python.exe" -m scripts.check_mcp
if ($LASTEXITCODE -ne 0) {
    throw "MCP server is not ready. Open another PowerShell window and run: .\run_mcp_server.ps1"
}

Write-Host "Opening SupportIQ at http://127.0.0.1:8501" -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m streamlit run ".\app.py" --server.address 127.0.0.1
