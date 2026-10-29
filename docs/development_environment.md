# Development Environment

## Programming Language

- Python: 3.11.9

## Development Software

- Visual Studio Code: 1.141.2
- Git: 2.48.1 (Windows)
- Windows PowerShell

## Python Dependencies

Python package names and pinned versions are maintained
in the project root `requirements.txt`.

## Environment Setup

Create a virtual environment:

```powershell
python -m venv .venv
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run tests:

```powershell
python -m pytest tests/ -v
```

## Project Conventions

- Relative paths for project files
- Simulation parameters in JSON
- NumPy complex128 for complex signals
- Git for version control
- Black for code formatting
