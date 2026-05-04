# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
#sys.path.insert(0,str(Path('./../../').resolve()))

import titan
print("TITAN MEMBERS:", dir(titan))
print("SUBMODULES:", [m for m in dir(titan) if not m.startswith("_")])

print(sys.path)
project = 'TITAN'
copyright = '2026, Fábio Morgado, Julie Graham, Tommy Williamson, Catarina Garbacz, Marco Fossati and contributors'
author = 'Fábio Morgado, Julie Graham, Tommy Williamson, Catarina Garbacz, Marco Fossati and contributors'
release = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
              "sphinx_immaterial",
              "sphinx_immaterial.apidoc.python.apigen"]

python_apigen_modules = {
      "titan": "api/titan/",
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_immaterial"
html_static_path = ['_static']
