"""
프로젝트 setup 자동화 스크립트.
"""

import pathlib

from setuptools import find_packages, setup
from datetime import datetime
import subprocess

here = pathlib.Path(__file__).parent
long_description = (
    (here / "docs/README.md").read_text() if (here / "docs/README.md").exists() else ""
)

with open("requirements.txt", encoding="utf-8") as f:
    all_reqs = f.read().splitlines()

requirements = [r for r in all_reqs if not r.startswith("-e ")]


def dynamic_version():
    date = datetime.today().strftime("%Y%m%d")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()

    except Exception:
        sha = "dev"
    return f"0.1.0+{date}-{sha}"


setup(
    name="JSG-ML-API",
    version=dynamic_version(),
    description="A ML API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Park Junha",
    author_email="jh01love00@gmail.com",
    url="https://github.com/TUK-JetSetGo/JSG-ML-API",
    packages=find_packages(exclude=["tests*", "docs"]),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "jsgmlapi=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
