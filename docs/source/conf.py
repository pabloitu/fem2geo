import os
import sys
from importlib.metadata import version as get_version

# If you use src/ layout (recommended)
sys.path.insert(0, os.path.abspath("../src"))

project = "fem2geo"
copyright = "2026, Pablo Iturrieta"
author = "Pablo Iturrieta"
release = get_version("fem2geo")  # from installed distribution metadata

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

github_username = "pabloitu"
github_repository = "fem2geo"

language = "en"
autosummary_generate = False
autoclass_content = "both"
suppress_warnings = ["autosummary", "autosummary.missing"]
templates_path = ["_templates"]
source_suffix = ".rst"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "default"
autodoc_typehints = "description"
autodoc_default_options = {
    'imported-members': False,
}
import pyvista
pyvista.OFF_SCREEN = True
pyvista.BUILDING_GALLERY = True
pyvista.set_plot_theme("document")
try:
    pyvista.start_xvfb()
except OSError:
    pass

sphinx_gallery_conf = {
    "examples_dirs": ["../../tutorials/examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"^.*\.py$",
    "image_scrapers": ("matplotlib", "pyvista"),
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
    "plot_gallery": "True",
    "thumbnail_size": (320, 224),
}
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

todo_include_todos = False
copybutton_prompt_text = "$ "
copybutton_only_copy_prompt_lines = False

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "titles_only": False,
}
html_logo = "_static/fem2geo_logo.png"

# Optional: set a logo if you add one under docs/_static/
# html_logo = "_static/fem2geo_logo.svg"

html_context = {
    "github_links": [
        ("Getting help", "https://github.com/pabloitu/fem2geo/issues"),
        ("Contributing", "https://github.com/pabloitu/fem2geo/blob/main/CONTRIBUTING.md"),
        ("License", "https://github.com/pabloitu/fem2geo/blob/main/LICENSE"),
        ("Source Code", "https://github.com/pabloitu/fem2geo"),
    ],
}

rst_epilog = """
.. raw:: html

    <hr />
    <div style="text-align: center;">
        <a href="https://github.com/pabloitu/fem2geo">GitHub</a> |
        <a href="https://pypi.org/project/fem2geo/">PyPI</a>
    </div>
"""
