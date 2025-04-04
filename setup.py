"""
프로젝트 setup 자동화 스크립트.
"""

import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent
long_description = (
    (here / "docs/README.md").read_text() if (here / "docs/README.md").exists() else ""
)

with open("requirements.txt", encoding="UTF-8") as f:
    requirements = f.read().splitlines()

setup(
    name="JSG-ML-API",
    version="0.1.0",
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
