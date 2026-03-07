# Illustrating the non-commutativity property of matrix multiplication: chaining linear transformations

This educational project demonstrates compound linear transformations on vectors and compares applying transformations in order `A → B` versus `B → A`.

## Project layout

```
linear_algebra/          ← project root (your application lives here)
├── pyproject.toml       ← packaging metadata
├── requirements.txt
├── performance_test.py  ← benchmark script
├── src/
│   └── linalg_3d/       ← installable library package
│       ├── __init__.py
│       ├── vector3d.py
│       ├── line_segment.py
│       └── transformations/
│           ├── __init__.py
│           ├── rotation.py
│           ├── reflection.py
│           ├── scaling.py
│           ├── shearing.py
│           └── translation.py
└── tests/
    ├── test_rotation.py
    ├── test_reflection.py
    ├── test_scaling.py
    ├── test_shearing.py
    └── test_translation.py
```

## Setup

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run tests

```powershell
python -m unittest discover -s tests -t .
```

## Run benchmark

```powershell
python performance_test.py
```

## App

Run the interactive visualization:

```powershell
python gui_app.py
```

## Notes

The reusable math library lives in `src/linalg_3d/` and is installed as an
editable package via `pip install -e .`. Application code (Tkinter UI, etc.)
can be placed at the project root and simply `import linalg_3d` — no
`sys.path` hacks needed.
