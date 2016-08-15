Apollo CAD
==========

Hi there!

You're looking at the _really_ early prototype of the `Apollo CAD` analysis. May the force be with you in your endevour!

## Development Environment

Most of the prototypes were done in ~~`iPython`~~ `Jupyter` notebooks, so having that is recommended. Further, make sure you work in a `virtual-environment` to let your imagination fly and the packages do not conflict.

To install the requirements, simply execute `pip install -r requirements.txt`. This will also install iPython packages.

### Virtual Environment Kernel

To allow `Jupyter` to work in your ~~safe place~~ virtual environment, create a Kernel spec file within your `jupyter` configs. To do so, follow these handy steps:

- Generate _some_ kernel spec. We'll be modifying it soon. Execute: `python -m ipykernel install --user --name apollo-cad`. This will install a kernel spec at `~/Library/Jupyter/kernels/apollo-cad`.
- Open `~/Library/Jupyter/kernels/apollo-cad/kernel.json` in your fav editor and simply update the first entry within `argv` prop so it points to the Python executable in your virtual environment instead. For lazy, type `which python` while the venv is activated.

## FAQ

> How do I ... ?

1. PEP 8 reference: https://www.python.org/dev/peps/pep-0008/
2. Python 3 docs: https://docs.python.org/3/

> Actually, no ... ?

1. Creating a virtualenvironment: https://virtualenvwrapper.readthedocs.io/en/latest/
2. Installing Jupyter: https://jupyter.readthedocs.io/en/latest/


