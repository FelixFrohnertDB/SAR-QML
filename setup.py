from setuptools import setup, find_packages


version = 0.1

install_requires = [
    "numpy",
    "scikit-learn",
    "pennylane",
    "jax",
    "optax",
    "pennylane-catalyst",
    "tqdm"
]

info = {
    "name": "sar_qml",
    "version": version,
    "author": "Felix Frohnert",
    "description": "A package for quantum machine learning for SAR data",
    "long_description": open('README.md').read(),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/FelixFrohnertDB/SAR-QML",
    "license": "Apache 2.0",
    "provides": ["sar_qml"],
    "install_requires": install_requires,
    "packages": find_packages(where='src'),
    "package_dir": {'': 'src'},
    "keywords": ["Quantum", "Machine Learning"],
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.12",
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Science/Research",
]

setup(classifiers=classifiers, **info)
