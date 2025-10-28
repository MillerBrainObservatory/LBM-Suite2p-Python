#!/bin/bash
# Compare documentation before/after uncommitted changes
cd "$(dirname "$0")"
uv run python compare_docs.py "$@"
