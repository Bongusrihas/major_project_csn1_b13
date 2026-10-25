"""One command: python run.py. Reuse the project environment or create it."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import venv


def main():
    root = Path(__file__).resolve().parent
    environment = root / '.venv'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        print('Creating the project Python environment...', flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    modules = ['numpy', 'pandas', 'sklearn', 'xgboost', 'torch', 'streamlit', 'plotly', 'matplotlib', 'joblib', 'scipy', 'filelock', 'pymupdf', 'psutil']
    check = 'import importlib.util,sys; sys.exit(any(importlib.util.find_spec(x) is None for x in ' + repr(modules) + '))'
    if subprocess.run([str(python), '-c', check], cwd=root).returncode:
        pip = subprocess.run([str(python), '-m', 'pip', '--version'], capture_output=True)
        if pip.returncode:
            subprocess.run([str(python), '-m', 'ensurepip', '--upgrade'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(root / 'requirements.txt')], check=True)
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')
    env['PYTHONUNBUFFERED'] = '1'
    subprocess.run([str(python), '-m', 'src.project.pipeline', *sys.argv[1:]], cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
