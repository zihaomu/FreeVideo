"""Create the container-path environment without replacing ROCm packages."""
import importlib.metadata as metadata
from pathlib import Path
import subprocess
import sys
import venv

root = Path('/data/experiments/freevideo-r9700')
environment = root / 'envs/rocm'
if sys.prefix != str(environment):
    venv.create(environment, system_site_packages=True, with_pip=True)
python = str(environment / 'bin/python')
# A venv created from another venv does not inherit that venv's packages.
# Keep the immutable image ROCm installation behind this environment's wheels.
image_site = '/opt/venv/lib/python3.12/site-packages'
(environment / 'lib/python3.12/site-packages/rocm-image.pth').write_text(image_site+'\n')
if image_site not in sys.path:
    sys.path.append(image_site)
constraints = root / 'envs/rocm-image-constraints.txt'
constraints.write_text(''.join(f'{p}=={metadata.version(p)}\n' for p in ('torch', 'torchvision', 'triton')))
subprocess.run([python, '-m', 'pip', 'install', '--no-cache-dir', '-c', str(constraints),
                '-c', '/workspace/scripts/amd/requirements-runtime.lock.txt',
                '-r', '/workspace/scripts/amd/requirements-runtime.txt'], check=True)
with (root / 'reports/runtime-pip-freeze.txt').open('w') as stream:
    subprocess.run([python, '-m', 'pip', 'freeze'], stdout=stream, check=True)
