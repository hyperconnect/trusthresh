from setuptools import find_packages
from setuptools import setup


setup(
    name="trusthresh",
    version="1.0.0",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "torch",
    ]
)
