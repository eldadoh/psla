from setuptools import setup, find_packages
from pathlib import Path

def get_requirements(req_file):
    with open(req_file) as f:
        packages = []
        for line in f:
            line = line.strip()
            # let's also ignore empty lines and comments
            if not line or line.startswith('#'):
                continue
            elif '--extra-index-url' in line:
                continue
            elif line.startswith('-r'):
                new_req_file = line.split(' ')[1]
                _path = Path(req_file).parent
                if _path is None:
                    path = ''
                new_packages = get_requirements(Path(_path, new_req_file))
                packages += new_packages
            else:
                packages.append(line)
    return packages


setup(
    name="psla",
    version="1.1",
    packages=find_packages(include=['psla', 'psla.*']),
    install_requires=get_requirements('requirements/requirements-dev.txt')
)