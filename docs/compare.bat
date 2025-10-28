@echo off
REM Compare documentation before/after uncommitted changes
cd /d "%~dp0"
uv run python compare_docs.py %*
