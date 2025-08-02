from setuptools import setup, find_packages

setup(
    name="tgea",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "cma",
        "pygad",
        "adversarial-robustness-toolbox",
    ],
    author="Pavlos Ntais",
    description="Transfer Guided Evolutionary Search (TGEA) codebase",
    python_requires=">=3.7",
)
