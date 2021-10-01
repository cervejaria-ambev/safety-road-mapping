# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------


import sys
import mock
import os

MOCK_MODULES = ['folium', 'folium.features', 'folium.map', 'numpy', 'openrouteservice',
                'pandas.core.frame', 'pandas', 'typeguard', 'typing', 'geopy', 'unidecode',
                'colour', 'pathlib', 'dotenv']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.MagicMock()


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'Safety Road Mapping'
copyright = '2021, Gabriel Aparecido Fonseca'
author = 'Gabriel Aparecido Fonseca'


# The full version, including alpha/beta/rc tags
def get_version():
    import re
    toml_file = open('../../pyproject.toml').read()
    return re.findall(r'(?<=version = ").*(?="\n)', toml_file)[0]


version = get_version()
release = version
print(f'Version: {release}')

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon',
              'sphinx.ext.viewcode']

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True
add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

autodoc_default_options = {"members": True, "undoc-members": True, "class-doc-from": True,
                           "special-members": "__init__", "private-members": True}
