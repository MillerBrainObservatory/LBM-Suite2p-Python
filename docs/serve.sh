#!/bin/bash
# Serve documentation on port 8888
cd "$(dirname "$0")"
uv run python serve_docs.py 8888
