# Documentation Setup Guide

This guide explains how to build and deploy the Sphinx documentation for the PINNs-QPINNs Heat Conduction Benchmark project.

## Project Structure

```
docs/
├── source/
│   ├── conf.py                       # Sphinx configuration
│   ├── index.rst                     # Main documentation index
│   ├── introduction.rst              # Introduction and overview
│   ├── theoretical_background.rst    # Mathematical foundations
│   ├── pinns_implementation.rst      # PINNs implementation details
│   ├── qpinns_implementation.rst     # QPINNs implementation details
│   ├── nsga2_optimizer.rst          # C++ NSGA-II optimizer
│   ├── experimental_results.rst      # Benchmark results
│   ├── api_reference.rst            # API documentation
│   ├── references.rst               # Scientific references
│   └── _static/                     # Static files (CSS, images)
├── build/                           # Generated documentation
├── Makefile                         # Build commands
├── Doxyfile                         # Doxygen configuration for C++
├── requirements-docs.txt            # Documentation dependencies
└── README_DOCS.md                   # This file
```

## Prerequisites

1. **Python 3.12+** with pip
2. **C++ compiler** with C++17 support
3. **Doxygen** (for C++ documentation)
4. **LaTeX** (for PDF generation, optional)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pinns-qpinns-benchmark.git
   cd pinns-qpinns-benchmark
   ```

2. **Install documentation dependencies:**
   ```bash
   pip install -r docs/requirements-docs.txt
   ```

3. **Install Doxygen (if not already installed):**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install doxygen graphviz
   
   # macOS
   brew install doxygen graphviz
   
   # Windows
   # Download from https://www.doxygen.nl/download.html
   ```

## Building the Documentation

### HTML Documentation

1. **Generate C++ documentation with Doxygen:**
   ```bash
   cd docs
   doxygen Doxyfile
   ```

2. **Build Sphinx HTML documentation:**
   ```bash
   make html
   ```

   The HTML documentation will be in `docs/build/html/`.

3. **View the documentation:**
   ```bash
   # Open in default browser
   open build/html/index.html  # macOS
   xdg-open build/html/index.html  # Linux
   start build/html/index.html  # Windows
   ```

### PDF Documentation

1. **Ensure LaTeX is installed:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install texlive-full
   
   # macOS
   brew install --cask mactex
   ```

2. **Build PDF:**
   ```bash
   make latexpdf
   ```

   The PDF will be in `docs/build/latex/`.

### Live Documentation Server

For development, you can use the live reload server:

```bash
make livehtml
```

This will start a server at `http://127.0.0.1:8000` that automatically rebuilds when you change source files.

## Adding New Content

### Adding a New Page

1. Create a new `.rst` file in `docs/source/`:
   ```rst
   New Section Title
   =================
   
   Your content here...
   ```

2. Add it to the table of contents in `index.rst`:
   ```rst
   .. toctree::
      :maxdepth: 2
      :caption: Contents:
      
      existing_page
      new_page  # Add your new page here
   ```

### Adding Mathematical Equations

Use LaTeX syntax within the documentation:

```rst
Inline math: :math:`\alpha = \frac{1}{2}`

Display math:

.. math::

   \frac{\partial u}{\partial t} = \alpha \nabla^2 u
```

### Adding Code Examples

```rst
.. code-block:: python
   :linenos:
   
   def example_function():
       """Example docstring"""
       return 42
```

### Adding References

In your `.rst` files:

```rst
According to Li et al. :cite:`Li2021`, the FNO approach...

.. bibliography:: references.bib
   :style: unsrt
```

## API Documentation

The API documentation is automatically generated from docstrings:

1. **For Python code:**
   ```python
   def compute_loss(self, u_pred, u_true):
       """Compute the loss between predictions and true values.
       
       Parameters
       ----------
       u_pred : torch.Tensor
           Predicted values
       u_true : torch.Tensor
           True values
           
       Returns
       -------
       torch.Tensor
           Computed loss value
       """
   ```

2. **For C++ code:**
   ```cpp
   /**
    * @brief Perform non-dominated sorting on population
    * 
    * @param population The population to sort
    * @return Vector of fronts
    */
   std::vector<std::vector<size_t>> non_dominated_sort(Population& population);
   ```

## Deployment

### GitHub Pages

1. **Build the documentation:**
   ```bash
   make html
   ```

2. **Create a `gh-pages` branch:**
   ```bash
   git checkout --orphan gh-pages
   git rm -rf .
   cp -r docs/build/html/* .
   echo "docs.yourdomain.com" > CNAME  # Optional custom domain
   git add .
   git commit -m "Deploy documentation"
   git push origin gh-pages
   ```

3. **Enable GitHub Pages in repository settings**

### Read the Docs

1. **Create account at** https://readthedocs.org

2. **Import your GitHub repository**

3. **Configure build:**
   - Python version: 3.12
   - Requirements file: `docs/requirements-docs.txt`
   - Sphinx configuration: `docs/source/conf.py`

## Troubleshooting

### Common Issues

1. **"Module not found" errors:**
   ```bash
   # Ensure the project is in Python path
   cd docs
   export PYTHONPATH=$PYTHONPATH:..
   ```

2. **LaTeX errors in PDF generation:**
   - Install missing LaTeX packages
   - Check for special characters in equations

3. **Doxygen XML not found:**
   - Run `doxygen Doxyfile` before building Sphinx
   - Check `breathe_projects` path in `conf.py`

### Clean Build

To start fresh:
```bash
make clean
rm -rf doxyxml/
```

## Documentation Standards

1. **Use Google-style docstrings** for Python code
2. **Use Doxygen comments** for C++ code
3. **Include examples** in docstrings where appropriate
4. **Add type hints** for better API documentation
5. **Reference papers** using proper citations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your documentation improvements
4. Build and test locally
5. Submit a pull request

## License

The documentation is licensed under the same terms as the main project (MIT License).