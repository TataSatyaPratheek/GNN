import os
from setuptools import setup, find_packages

setup(
    name="processmine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.23.0",
        "torch>=1.8.0",  # Core PyTorch dependency
    ],
    extras_require={
        # Optional dependencies that are required for specific features
        "gnn": [
            "torch-geometric>=2.0.0",
            "torch-scatter>=2.0.0",
            "torch-sparse>=0.6.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=4.14.0",
            "networkx>=2.5.0",
        ],
        "all": [
            "torch-geometric>=2.0.0",
            "torch-scatter>=2.0.0",
            "torch-sparse>=0.6.0",
            "seaborn>=0.11.0",
            "plotly>=4.14.0",
            "networkx>=2.5.0",
            "dask>=2022.1.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "coverage>=5.3.0",
        ],
    },
    author="ProcessMine Team",
    author_email="info@processmine.com",
    description="Memory-efficient process mining with Graph Neural Networks and LSTM models",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/processmine/processmine",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)