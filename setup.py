"""Setup script for Hive Mind."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hive-mind",
    version="0.1.0",
    author="Hive Mind Team",
    author_email="team@hive-mind.ai",
    description="Multi-model conversation framework with collective intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hive-mind/hive-mind",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.11.0",
            "isort>=5.12.0",
            "mypy>=1.7.1",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hive-mind=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "hive_mind": ["*.yaml", "*.yml", "*.json"],
    },
)