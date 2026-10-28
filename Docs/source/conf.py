# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

"""conf module."""
project = 'TITAN'
copyright = '2026, TITAN Contributors'
author = 'Fabio Morgado, Julie Graham, Tommy Williamson,\n Catarina Garbacz, Marco Fossati & Contributors'
release = 'v0.2 "Atlas"'

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc"]
#extensions.append("sphinx_breeze_theme")
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'breeze'
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_theme_options = {
    "dark_logo": "titan_icon_dark.png",
    "light_logo": "titan_icon_light.png",

}


#html_logo = 'titan_icon_dark.png'

