@echo off
REM Serve documentation on port 8888
cd /d "%~dp0"
uv run python serve_docs.py 8888
