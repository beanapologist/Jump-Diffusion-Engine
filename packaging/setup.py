from setuptools import setup
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


setup(
    name="jump-diffusion-engine",
    version="0.1.0",
    description="Jump diffusion simulation engine for stochastic control systems.",
    py_modules=["engine"],
    package_dir={"": str(PROJECT_ROOT)},
    install_requires=[
        "numpy>=1.23,<3.0",
        "scipy>=1.10,<2.0",
        "matplotlib>=3.7,<4.0",
    ],
)
