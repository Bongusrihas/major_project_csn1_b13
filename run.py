"""One command: python run.py. Reuse the project environment or create it."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv


def check_device():
    """Show the installed backend and verify that the selected device executes."""
    import torch
    print(f'PyTorch: {torch.__version__}', flush=True)
    print(f'CUDA runtime in PyTorch: {torch.version.cuda or "none (CPU-only build)"}', flush=True)
    if torch.cuda.is_available():
        print(f'Training device: cuda — {torch.cuda.get_device_name(0)}', flush=True)
        sample = torch.randn(32, 32, device='cuda', requires_grad=True)
        (sample @ sample.T).square().mean().backward()
        torch.cuda.synchronize()
        print('CUDA forward/backward check: passed', flush=True)
    else:
        print('Training device: cpu', flush=True)
        if torch.version.cuda is None:
            print('This PyTorch build has no CUDA support. In an updated checkout, run uv sync --locked.', flush=True)
        else:
            print('CUDA is installed, but no usable NVIDIA GPU/driver is visible. Check nvidia-smi.', flush=True)
    print('Preprocessing, plots and the residual boosting head use CPU. The neural network uses the training device.', flush=True)


def main():
    root = Path(__file__).resolve().parent
    environment = root / '.venv'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    uv = shutil.which('uv')
    if not python.exists():
        print('Creating the project Python environment...', flush=True)
        if uv:
            subprocess.run([uv, 'sync', '--locked', '--project', str(root)], check=True)
        else:
            venv.EnvBuilder(with_pip=True).create(environment)
    modules = ['numpy', 'pandas', 'sklearn', 'xgboost', 'torch', 'streamlit', 'plotly', 'matplotlib', 'joblib', 'scipy', 'filelock', 'pymupdf', 'psutil']
    check = 'import importlib.util,sys; sys.exit(any(importlib.util.find_spec(x) is None for x in ' + repr(modules) + '))'
    if subprocess.run([str(python), '-c', check], cwd=root).returncode:
        if uv:
            # Respect the locked CUDA source instead of replacing it with PyPI's CPU wheel.
            subprocess.run([uv, 'sync', '--locked', '--project', str(root)], check=True)
        else:
            pip = subprocess.run([str(python), '-m', 'pip', '--version'], capture_output=True)
            if pip.returncode:
                subprocess.run([str(python), '-m', 'ensurepip', '--upgrade'], check=True)
            subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(root / 'requirements.txt')], check=True)
    if '--check-device' in sys.argv[1:]:
        subprocess.run([str(python), '-c', 'from run import check_device; check_device()'], cwd=root, check=True)
        return
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')
    env['PYTHONUNBUFFERED'] = '1'
    subprocess.run([str(python), '-m', 'src.project.pipeline', *sys.argv[1:]], cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
