#!/usr/bin/env python
"""
build and serve documentation locally.

usage:
    uv run python scripts/build_docs.py          # build and open
    uv run python scripts/build_docs.py --clean  # clean build first
    uv run python scripts/build_docs.py --no-open  # build without opening
"""
import argparse
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path


def copy_notebooks(docs_dir: Path, notebooks_dir: Path):
    """copy notebooks from demos/notebooks/ to docs/ for sphinx-build."""
    notebooks = [
        "user_guide.ipynb",
        "quickstart.ipynb",
        "projections.ipynb",
    ]

    copied = []
    for nb in notebooks:
        src = notebooks_dir / nb
        dst = docs_dir / nb
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(nb)
            print(f"  copied {nb}")
        else:
            print(f"  skipped {nb} (not found)")

    return copied


def build_docs(root: Path, clean: bool = False):
    """build sphinx documentation."""
    docs_dir = root / "docs"
    notebooks_dir = root / "demos" / "notebooks"
    build_dir = docs_dir / "_build"
    html_dir = build_dir / "html"

    # clean if requested
    if clean and build_dir.exists():
        print("cleaning build directory...")
        shutil.rmtree(build_dir)

    # copy notebooks
    print("\ncopying notebooks from demos/notebooks/ to docs/...")
    copy_notebooks(docs_dir, notebooks_dir)

    # build html
    print("\nbuilding documentation...")
    result = subprocess.run(
        [
            sys.executable, "-m", "sphinx",
            "-b", "html",
            str(docs_dir),
            str(html_dir),
        ],
        cwd=root,
    )

    if result.returncode != 0:
        print("\n[ERROR] documentation build failed")
        return None

    return html_dir / "index.html"


def main():
    parser = argparse.ArgumentParser(description="build documentation")
    parser.add_argument("--clean", "-c", action="store_true", help="clean build directory first")
    parser.add_argument("--no-open", action="store_true", help="don't open browser")
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    index_path = build_docs(root, clean=args.clean)

    if index_path is None:
        sys.exit(1)

    print(f"\n[SUCCESS] documentation built: {index_path}")

    if not args.no_open:
        print("opening in browser...")
        webbrowser.open(index_path.as_uri())


if __name__ == "__main__":
    main()
