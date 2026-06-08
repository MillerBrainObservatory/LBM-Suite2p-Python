# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join("..")))
sys.path.insert(0, os.path.abspath(os.path.join("..", "lbm_suite2p_python")))

# copy notebooks from demos/notebooks/ to docs/ before building
def setup(app):
    """copy notebooks from demos/notebooks/ to docs/ before building."""
    docs_dir = Path(__file__).parent
    notebooks_dir = docs_dir.parent / "demos" / "notebooks"

    notebooks = ["user_guide.ipynb", "quickstart.ipynb", "projections.ipynb"]

    for nb in notebooks:
        src = notebooks_dir / nb
        dst = docs_dir / nb
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {src.name} to docs/")
        else:
            print(f"Warning: {nb} not found at {src}")

from lbm_suite2p_python import __version__

project = "LBM-Suite2p-Python"
author = ""

copyright = "2024, Elizabeth R. Miller Brain Observatory | The Rockefeller University. All Rights Reserved"
release = __version__

exclude_patterns = ["Thumbs.db", ".DS_Store", "_build*"]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "html_image",
]

extensions = [
    "sphinx.ext.autodoc",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "sphinx_tippy",
]
bibtex_bibfiles = ["refs.bib"]


source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

nb_execution_mode = "off"

myst_admonition_enable = True
myst_amsmath_enable = True
myst_html_img_enable = True
myst_url_schemes = ("http", "https", "mailto")
myst_heading_anchors = 3  # generate GitHub-style anchors so in-page [text](#slug) links resolve

images_config = {"cache_path": "./_images/"}

templates_path = ["_templates"]

# A shorter title for the navigation bar.  Default is the same as html_title.
html_title = "LBM-Suite2p-Python"

html_theme = "sphinx_book_theme"
html_logo = "./_static/logo_suit2p.png"
html_favicon = "./_static/icon_suite2p.svg"

html_short_title = "LBM Suite2p Pipeline"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_copy_source = True
html_file_suffix = ".html"
# html_use_modindex = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.9", None),
    "numpydoc": ("https://numpydoc.readthedocs.io/en/latest", None),
    "mbo": (
        "https://millerbrainobservatory.github.io",
        None,
    ),
    "mbo_utilities": (
        "https://millerbrainobservatory.github.io/mbo_utilities",
        None,
    ),
    "lbm_suite2p_python": (
        "https://millerbrainobservatory.github.io/LBM-Suite2p-Python",
        None,
    ),
    "suite2p": ("https://suite2p.readthedocs.io/en/latest/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}


intersphinx_disabled_reftypes = ["*"]

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/MillerBrainObservatory/LBM-Suite2p-Python/",
    "repository_branch": "master",
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "show_toc_level": 2,
    "navbar_align": "content",
    "icon_links": [
        {
            "name": "MBO User Hub",
            "url": "https://millerbrainobservatory.github.io/",
            "icon": "_static/icon_mbo_home.png",
            "type": "local",
        },
        {
            "name": "MBO Github",
            "url": "https://github.com/MillerBrainObservatory/",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Connect with MBO",
            "url": "https://mbo.rockefeller.edu/contact/",
            "icon": "fa-regular fa-address-card",
            "type": "fontawesome",
        },
    ],
}
