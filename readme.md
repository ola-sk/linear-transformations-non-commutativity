# Illustrating the non-commutativity property of matrix multiplication: chaining linear transformations

This educational project demonstrates compound linear transformations on
vectors and compares applying transformations in order `A → B` versus `B → A`.

A Tkinter GUI lets the user choose two 3D transformations (rotation, scaling,
shearing, reflection, or identity), apply them cumulatively to a figure, and
observe how the two orderings diverge over successive steps.

## Quick start

> **Prerequisites:** Python ≥ 3.12 with Tkinter (included in the standard
> Windows / macOS CPython installer; on Linux run
> `sudo apt install python3-tk` if needed).

```bash
pip install -r requirements.txt   # install dependencies (numpy, matplotlib)
python gui_app.py                 # launch the GUI
```

## Running the tests

```bash
python -m unittest discover -s tests -t .
```

## Project layout

```
linear_algebra/
├── pyproject.toml           ← packaging metadata & dependency list
├── requirements.txt         ← runtime dependencies
├── readme.md
├── gui_app.py               ← Tkinter GUI application
├── linalg_3d/               ← reusable library package
│   ├── __init__.py
│   ├── vector3d.py
│   ├── line_segment.py
│   ├── figures/
│   │   ├── __init__.py
│   │   ├── cube.py
│   │   ├── fish.py
│   │   └── star.py
│   └── transformations/
│       ├── __init__.py
│       ├── rotation.py
│       ├── reflection.py
│       ├── scaling.py
│       ├── shearing.py
│       └── translation.py
└── tests/
    ├── __init__.py
    ├── test_vector3d.py
    ├── test_line_segment.py
    ├── test_figures.py
    ├── test_rotation.py
    ├── test_reflection.py
    ├── test_scaling.py
    ├── test_shearing.py
    └── test_translation.py
```

## Role of `pyproject.toml`

`pyproject.toml` is the standard Python packaging configuration file
([PEP 621](https://peps.python.org/pep-0621/)).  In this project it:

* declares the build system (`setuptools`);
* lists the package name, version, Python version constraint, and runtime
  dependencies (`numpy`, `matplotlib`);
* defines optional `dev` dependencies (e.g. `pytest`) and `pytest`
  configuration.

It also enables installing the library into other projects with
`pip install -e .` or directly from the repository URL.

## Development setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python gui_app.py

# 4. Run tests
python -m unittest discover -s tests -t .
```
