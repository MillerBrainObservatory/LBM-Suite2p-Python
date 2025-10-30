#!/usr/bin/env python
"""
Build and serve documentation before/after uncommitted changes.

This script:
1. Builds current (modified) docs → serves on port 8888
2. Builds original (HEAD) docs → serves on port 8889
3. Creates a comparison page at http://localhost:8890

Usage:
    python compare_docs.py [--no-build] [--ports PORT1,PORT2,PORT3] [--skip-original]

Options:
    --no-build           Skip rebuilding docs (use existing builds)
    --skip-original      Only serve current docs (faster, no git worktree)
    --ports PORT1,PORT2,PORT3  Custom ports (default: 8888,8889,8890)
"""
import subprocess
import shutil
import sys
import threading
import http.server
import socketserver
from pathlib import Path
import argparse
import tempfile


def run_command(cmd, cwd=None, check=True, verbose=True):
    """Run shell command and return result."""
    if verbose:
        print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=False,
                          capture_output=True, text=True)
    if result.returncode != 0:
        if verbose:
            print(f"❌ Command failed with exit code {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr}")
            if result.stdout:
                print(f"Standard output:\n{result.stdout}")
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


def build_current_docs(docs_dir):
    """Build current (modified) documentation."""
    print("\n" + "="*60)
    print("Building CURRENT docs (with uncommitted changes)...")
    print("="*60)

    build_dir = docs_dir / "_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

    result = run_command("uv run make html", cwd=docs_dir, check=False)
    if result.returncode != 0:
        print("❌ Failed to build current docs")
        sys.exit(1)

    print("✓ Current docs built successfully\n")
    return docs_dir / "_build" / "html"


def build_original_docs(repo_root, docs_dir):
    """Build original (HEAD) documentation in a temporary location."""
    print("\n" + "="*60)
    print("Building ORIGINAL docs (from git HEAD)...")
    print("="*60)

    # Create temporary directory for original docs
    temp_dir = docs_dir / "_build_before"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    # Use git worktree to checkout HEAD in temp location
    worktree_dir = temp_dir / "worktree"
    print(f"Creating git worktree at {worktree_dir}...")

    result = run_command(f'git worktree add "{worktree_dir}" HEAD', cwd=repo_root, check=False)

    if result.returncode != 0:
        print("\n⚠️  Warning: Could not create worktree.")
        print("Trying git archive method as fallback...")

        # Fallback: use git archive
        worktree_dir.mkdir(parents=True, exist_ok=True)
        archive_result = run_command(
            f'git archive HEAD | tar -x -C "{worktree_dir}"',
            cwd=repo_root,
            check=False
        )

        if archive_result.returncode != 0:
            print("\n❌ Failed to create git archive.")
            print("You can use --skip-original to only serve current docs.")
            sys.exit(1)

    # Build docs in the worktree
    worktree_docs = worktree_dir / "docs"

    print(f"\nBuilding docs in worktree: {worktree_docs}")
    print("This may take a minute...")

    # Clean any existing build in worktree
    worktree_build = worktree_docs / "_build"
    if worktree_build.exists():
        shutil.rmtree(worktree_build)

    build_result = run_command("uv run make html", cwd=worktree_docs, check=False)

    if build_result.returncode != 0:
        print("\n❌ Failed to build original docs in worktree")
        print("\nPossible reasons:")
        print("  - Missing dependencies in git HEAD version")
        print("  - Sphinx configuration errors")
        print("  - Missing files")
        print("\nYou can use --skip-original to only serve current docs.")

        # Clean up worktree before exiting
        if (worktree_dir / ".git").exists():
            run_command(f'git worktree remove "{worktree_dir}" --force',
                       cwd=repo_root, check=False, verbose=False)

        sys.exit(1)

    # Copy built HTML to our temp location
    original_html = temp_dir / "html"
    worktree_html = worktree_docs / "_build" / "html"

    if not worktree_html.exists():
        print(f"\n❌ Built HTML not found at {worktree_html}")
        sys.exit(1)

    shutil.copytree(worktree_html, original_html)

    # Clean up worktree
    print("Cleaning up git worktree...")
    if (worktree_dir / ".git").exists():
        run_command(f'git worktree remove "{worktree_dir}" --force',
                   cwd=repo_root, check=False, verbose=False)

    print("✓ Original docs built successfully\n")
    return original_html


def create_comparison_page(port1, port2, output_dir):
    """Create an HTML comparison page with side-by-side iframes."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation Comparison</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1e1e1e;
            color: #fff;
        }}
        .header {{
            background: #252526;
            padding: 15px 20px;
            border-bottom: 1px solid #3e3e42;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            font-size: 18px;
            font-weight: 600;
        }}
        .controls {{
            display: flex;
            gap: 15px;
        }}
        .btn {{
            background: #0e639c;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: background 0.2s;
        }}
        .btn:hover {{
            background: #1177bb;
        }}
        .btn.danger {{
            background: #d13438;
        }}
        .btn.danger:hover {{
            background: #e81123;
        }}
        .container {{
            display: flex;
            height: calc(100vh - 60px);
        }}
        .panel {{
            flex: 1;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #3e3e42;
        }}
        .panel:last-child {{
            border-right: none;
        }}
        .panel-header {{
            background: #2d2d30;
            padding: 10px 15px;
            border-bottom: 1px solid #3e3e42;
            font-size: 13px;
            font-weight: 500;
        }}
        .panel-header.before {{
            background: #3e2723;
        }}
        .panel-header.after {{
            background: #1b3a1f;
        }}
        iframe {{
            flex: 1;
            border: none;
            background: white;
        }}
        .toggle-layout {{
            display: flex;
            gap: 5px;
        }}
        .toggle-layout button {{
            padding: 6px 12px;
            background: #3e3e42;
            border: none;
            color: #ccc;
            cursor: pointer;
            border-radius: 3px;
            font-size: 12px;
        }}
        .toggle-layout button.active {{
            background: #0e639c;
            color: white;
        }}
        .container.vertical {{
            flex-direction: column;
        }}
        .status {{
            font-size: 12px;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 Documentation Comparison</h1>
        <div class="controls">
            <div class="toggle-layout">
                <button id="splitBtn" class="active" onclick="setLayout('horizontal')">Split</button>
                <button id="verticalBtn" onclick="setLayout('vertical')">Stack</button>
                <button id="beforeBtn" onclick="setLayout('before')">Before Only</button>
                <button id="afterBtn" onclick="setLayout('after')">After Only</button>
            </div>
            <button class="btn" onclick="syncFrames()">🔄 Sync</button>
            <button class="btn" onclick="location.reload()">↻ Reload</button>
        </div>
    </div>
    <div class="container" id="container">
        <div class="panel" id="beforePanel">
            <div class="panel-header before">
                BEFORE (git HEAD) → <a href="http://localhost:{port2}" target="_blank" style="color: #4fc3f7;">localhost:{port2}</a>
            </div>
            <iframe id="beforeFrame" src="http://localhost:{port2}"></iframe>
        </div>
        <div class="panel" id="afterPanel">
            <div class="panel-header after">
                AFTER (current changes) → <a href="http://localhost:{port1}" target="_blank" style="color: #81c784;">localhost:{port1}</a>
            </div>
            <iframe id="afterFrame" src="http://localhost:{port1}"></iframe>
        </div>
    </div>

    <script>
        let currentLayout = 'horizontal';

        function setLayout(layout) {{
            const container = document.getElementById('container');
            const beforePanel = document.getElementById('beforePanel');
            const afterPanel = document.getElementById('afterPanel');
            const buttons = document.querySelectorAll('.toggle-layout button');

            buttons.forEach(btn => btn.classList.remove('active'));

            currentLayout = layout;

            switch(layout) {{
                case 'horizontal':
                    container.className = 'container';
                    beforePanel.style.display = 'flex';
                    afterPanel.style.display = 'flex';
                    document.getElementById('splitBtn').classList.add('active');
                    break;
                case 'vertical':
                    container.className = 'container vertical';
                    beforePanel.style.display = 'flex';
                    afterPanel.style.display = 'flex';
                    document.getElementById('verticalBtn').classList.add('active');
                    break;
                case 'before':
                    container.className = 'container';
                    beforePanel.style.display = 'flex';
                    afterPanel.style.display = 'none';
                    document.getElementById('beforeBtn').classList.add('active');
                    break;
                case 'after':
                    container.className = 'container';
                    beforePanel.style.display = 'none';
                    afterPanel.style.display = 'flex';
                    document.getElementById('afterBtn').classList.add('active');
                    break;
            }}
        }}

        function syncFrames() {{
            const beforeFrame = document.getElementById('beforeFrame');
            const afterFrame = document.getElementById('afterFrame');

            try {{
                const currentPath = afterFrame.contentWindow.location.pathname;
                beforeFrame.src = 'http://localhost:{port2}' + currentPath;
                console.log('Synced to:', currentPath);
            }} catch(e) {{
                console.warn('Cannot sync frames due to CORS. Navigate manually.');
                alert('Frames are on different origins. Please navigate manually.');
            }}
        }}

        // Auto-reload frames when server restarts
        setInterval(() => {{
            const beforeFrame = document.getElementById('beforeFrame');
            const afterFrame = document.getElementById('afterFrame');
            // Check if iframes are still accessible
            try {{
                beforeFrame.contentWindow.location.href;
                afterFrame.contentWindow.location.href;
            }} catch(e) {{
                // Frame might be reloading
            }}
        }}, 5000);
    </script>
</body>
</html>
"""

    output_file = output_dir / "index.html"
    output_file.write_text(html_content, encoding="utf-8")
    print(f"✓ Comparison page created at {output_file}")
    return output_file


def serve_directory(directory, port):
    """Serve a directory on specified port."""
    import os

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format, *args):
            # Suppress request logs
            pass

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"✓ Serving on http://localhost:{port}")
        httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Compare documentation before/after changes")
    parser.add_argument("--no-build", action="store_true", help="Skip rebuilding docs")
    parser.add_argument("--skip-original", action="store_true",
                       help="Only serve current docs (skip git HEAD build)")
    parser.add_argument("--ports", default="8888,8889,8890",
                       help="Ports for current,original,comparison (default: 8888,8889,8890)")
    args = parser.parse_args()

    ports = [int(p) for p in args.ports.split(",")]
    port_current, port_original, port_comparison = ports

    repo_root = Path(__file__).parent.parent
    docs_dir = Path(__file__).parent

    if not args.no_build:
        # Build current version
        current_html = build_current_docs(docs_dir)

        # Build original version (unless skipped)
        if not args.skip_original:
            try:
                original_html = build_original_docs(repo_root, docs_dir)
            except Exception as e:
                print(f"\n❌ Error building original docs: {e}")
                print("\nContinuing with only current docs...")
                print("Use --skip-original to suppress this message.")
                args.skip_original = True
        else:
            print("\n⏭️  Skipping original docs build (--skip-original)")
    else:
        current_html = docs_dir / "_build" / "html"
        original_html = docs_dir / "_build_before" / "html"

        if not current_html.exists():
            print(f"❌ Error: {current_html} not found. Remove --no-build flag.")
            sys.exit(1)

        if not args.skip_original and not original_html.exists():
            print(f"⚠️  Warning: {original_html} not found.")
            print("Continuing with only current docs...")
            args.skip_original = True

    # Start servers in background threads
    print("\n" + "="*60)
    print("Starting documentation servers...")
    print("="*60 + "\n")

    threads = []

    # Always serve current docs
    threads.append(
        threading.Thread(target=serve_directory, args=(current_html, port_current), daemon=True)
    )

    # Serve original docs if available
    if not args.skip_original:
        # Create comparison page
        comparison_dir = docs_dir / "_build_comparison"
        comparison_dir.mkdir(exist_ok=True)
        create_comparison_page(port_current, port_original, comparison_dir)

        threads.append(
            threading.Thread(target=serve_directory, args=(original_html, port_original), daemon=True)
        )
        threads.append(
            threading.Thread(target=serve_directory, args=(comparison_dir, port_comparison), daemon=True)
        )

    for t in threads:
        t.start()

    print("\n" + "="*60)
    if args.skip_original:
        print("📚 Documentation Server Ready!")
        print("="*60)
        print(f"\n🌟 Current docs: http://localhost:{port_current}")
    else:
        print("📚 Documentation Comparison Ready!")
        print("="*60)
        print(f"\n🌟 Main comparison view: http://localhost:{port_comparison}")
        print(f"\n   AFTER (current):      http://localhost:{port_current}")
        print(f"   BEFORE (git HEAD):    http://localhost:{port_original}")
    print("\n" + "="*60)
    print("\nPress Ctrl+C to stop all servers\n")

    try:
        # Keep main thread alive
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down all servers...")
        sys.exit(0)


if __name__ == "__main__":
    main()
