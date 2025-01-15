#!/usr/bin/env python3

import setuptools
import versioneer
from pathlib import Path

install_deps = [
    # "tifffile",
    # "numpy",
    # "numba>=0.57.0",
    # "scipy>=1.9.0",
    "fastplotlib[notebook]",
    # "scanreader @ git+https://github.com/atlab/scanreader.git@master#egg=scanreader",
    # "matplotlib",
    # "lcp-mc",
    # "fabric",
    "dask",
    "zarr",
    "jupyterlab",
    "suite2p[all]"
]

extras_require = {
    "docs": [
        "sphinx>=6.1.3",
        "docutils>=0.19",
        "nbsphinx",
        "numpydoc",
        "sphinx-autodoc2",
        "sphinx_gallery",
        "sphinx-togglebutton",
        "sphinx-copybutton",
        "sphinx_book_theme",
        "pydata_sphinx_theme",
        "sphinx_design",
        "sphinxcontrib-images",
        "sphinxcontrib-video",
        "sphinx_tippy",
        "myst_nb",
    ],
}

with open(Path(__file__).parent / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lbm_suite2p_python",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Light Beads Microscopy 2P Calcium Imaging Pipeline.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    license="BSD-3-Clause",
    url="https://github.com/millerbrainobservatory/LBM-Suite2p-Python",
    keywords="Pipeline Numpy Microscopy ScanImage Suite2p tiff",
    install_requires=install_deps,
    extras_require=extras_require,
    packages=setuptools.find_packages(exclude=["data", "data.*"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
    ],
    entry_points={
        "console_scripts": [
            "lsp = lbm_suite2p_python.__main__:main",
        ]
    },
)

