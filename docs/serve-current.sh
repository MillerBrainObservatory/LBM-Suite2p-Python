#!/bin/bash
# Serve ONLY current docs (skip git HEAD comparison)
# This is faster and doesn't require git worktree
cd "$(dirname "$0")"
uv run python compare_docs.py --skip-original
