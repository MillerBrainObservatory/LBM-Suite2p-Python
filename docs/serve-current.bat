@echo off
REM Serve ONLY current docs (skip git HEAD comparison)
REM This is faster and doesn't require git worktree
cd /d "%~dp0"
uv run python compare_docs.py --skip-original
