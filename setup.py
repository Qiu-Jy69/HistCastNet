#!/usr/bin/env python
import io
import os
import re
from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version("src", "histcastnet", "__init__.py")

requirements = [
    "h5py>=2.10.0",
    "matplotlib",
    "packaging",
    "Pillow",
    "numpy",
    "omegaconf",
    "pandas",
    "requests",
    "pytorch-lightning>=1.8",
    "scipy",
    "torch>=1.10",
    "torchmetrics",
    "tqdm",
]

setup(
    # Metadata
    name="histcastnet",
    version=VERSION,
    python_requires=">=3.9",
    description="State-guided historical retrieval for precipitation nowcasting",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    # Package info
    packages=find_packages(
        where="src",
        exclude=(
            "tests",
            "scripts",
        ),
    ),
    package_dir={"": "src"},
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
)
