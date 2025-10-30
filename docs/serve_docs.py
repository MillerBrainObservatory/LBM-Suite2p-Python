#!/usr/bin/env python
"""
Serve Sphinx documentation on a local port.

Usage:
    python serve_docs.py [port] [build_dir]

Examples:
    python serve_docs.py 8888                    # Serve on port 8888
    python serve_docs.py 8889 _build_before/html  # Serve old version on 8889
"""
import http.server
import socketserver
import sys
from pathlib import Path


def serve_docs(port: int = 8888, build_dir: str = "_build/html"):
    """Serve documentation on specified port."""
    docs_path = Path(__file__).parent / build_dir

    if not docs_path.exists():
        print(f"Error: Documentation directory not found: {docs_path}")
        print(f"Run 'uv run make html' first to build the docs.")
        sys.exit(1)

    # Change to the documentation directory
    import os
    os.chdir(docs_path)

    Handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"\n{'='*60}")
        print(f"Serving documentation at: http://localhost:{port}")
        print(f"Directory: {docs_path.absolute()}")
        print(f"{'='*60}\n")
        print("Press Ctrl+C to stop the server\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            httpd.shutdown()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    build_dir = sys.argv[2] if len(sys.argv) > 2 else "_build/html"
    serve_docs(port, build_dir)
