# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'PINNs-QPINNs Heat Conduction Benchmark'
copyright = '2025, Research Team'
author = 'Research Team'

# The full version, including alpha/beta/rc tags
release = '1.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
    'sphinx.ext.duration',
    'sphinx_rtd_theme',
    'sphinx.ext.autosummary',
    'breathe'  # For Doxygen integration
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ['_static']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Theme options
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# -- Options for LaTeX output ------------------------------------------------

latex_engine = 'pdflatex'
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{physics}
\usepackage{algorithm}
\usepackage{algorithmic}
''',
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Breathe configuration for Doxygen
breathe_projects = {
    "nsga2_optimizer": "../doxyxml",
}
breathe_default_project = "nsga2_optimizer"

# Todo extension
todo_include_todos = True

# MathJax configuration (version 3)
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$', '$'], ['\\[', '\\]']],
        'processEscapes': True,
        'macros': {
            'bm': ['{\\boldsymbol{#1}}', 1],
            'abs': ['\\left|#1\\right|', 1],
            'norm': ['\\left\\|#1\\right\\|', 1],
        }
    },
    'options': {
        'skipHtmlTags': ['script', 'noscript', 'style', 'textarea', 'pre'],
        'processHtmlClass': 'tex2jax_process',
    },
}
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'


# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}


# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'pennylane': ('https://docs.pennylane.ai/en/stable/', None),
}


# Autosummary settings
autosummary_generate = True

# -- Custom setup ------------------------------------------------------------

def setup(app):
    app.add_css_file('custom.css')