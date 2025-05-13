"""
프로젝트 setup 자동화 스크립트.
"""

import pathlib
import subprocess
from datetime import datetime

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent
long_description = (
    (here / "docs/README.md").read_text() if (here / "docs/README.md").exists() else ""
)

with open("requirements.txt", encoding="UTF-8") as f:
    requirements = f.read().splitlines()


def get_version():
    """
    base_version+YYYYMMDD-gitSHA. 형식으로 버전 생성
    SHA가 유효하지 않으면 실패.
    """
    base_version = "0.1.0"
    date = datetime.now().strftime("%Y%m%d")
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        sha = "dev"
    return f"{base_version}+{date}-{sha}"


setup(
    name="JSG-ML-API",
    version=get_version(),
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
