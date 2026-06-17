from setuptools import setup


setup(
    name="jump-diffusion-engine",
    version="0.1.0",
    description="Jump diffusion simulation engine for stochastic control systems.",
    py_modules=["engine"],
    package_dir={"": ".."},
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
)
