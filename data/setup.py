#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup.py
========
EAT-Lab Framework - Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f 
                if line.strip() and not line.startswith('#') 
                and not line.startswith('-r')]

setup(
    name="eat-lab-framework",
    version="3.0.0",
    author="EAT-Lab Collaborative",
    author_email="contact@eat-lab.org",
    description="Framework completo para análise semântica e causal de LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eat-lab/framework",
    project_urls={
        "Bug Tracker": "https://github.com/eat-lab/framework/issues",
        "Documentation": "https://eat-lab.readthedocs.io",
        "Source Code": "https://github.com/eat-lab/framework",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "minimal": read_requirements("requirements-minimal.txt"),
        "all": read_requirements("requirements-dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "eat-score=eat_lab.cli:main_score",
            "eat-trace=eat_lab.cli:main_trace",
            "eat-density=eat_lab.cli:main_density",
        ],
    },
    include_package_data=True,
    package_data={
        "eat_lab": ["data/*.json", "configs/*.yaml"],
    },
    zip_safe=False,
)
