import setuptools
from setuptools import Extension

import numpy as np

# Cython is optional: if available at build time, compile accelerated extensions.
# If not available, the pure Python versions in Pynite/cython/*.py will be used.
try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - when Cython is missing we build without extensions
    cythonize = None

with open("README.md", "r") as fh:
    long_description = fh.read()


def build_extensions():
    """Build Cython extensions if Cython is available, otherwise return empty list."""
    if cythonize is None:
        return []

    extensions = [
        Extension(
            "Pynite.cython.sparse",
            ["Pynite/cython/sparse.py"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "Pynite.cython.kernels",
            ["Pynite/cython/kernels.py"],
            include_dirs=[np.get_include()],
        ),
    ]
    return cythonize(
        extensions,
        compiler_directives={"language_level": "3", "boundscheck": False, "wraparound": False},
    )


setuptools.setup(
    name="PyNiteFEA",
    version="1.5.0",
    author="D. Craig Brinck, PE, SE",
    author_email="Building.Code@outlook.com",
    description="A simple elastic 3D structural finite element library for Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JWock82/Pynite.git",
    packages=setuptools.find_packages(include=["Pynite", "Pynite.*"]),
    package_data={"Pynite": ["*html", "*.css", "*Full Logo No Buffer - Transparent.png"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "PrettyTable",
        "scipy",
        "pip>=25.0.1",
    ],
    extras_require={
        "all": [
            "Ipython",
            "vtk",
            "pyvista[all,trame]",
            "trame_jupyter_extension",
            "ipywidgets",
            "pdfkit",
            "Jinja2",
            "matplotlib",
        ],
        "vtk": ["IPython", "vtk"],
        "pyvista": ["pyvista[all,trame]", "trame_jupyter_extension", "ipywidgets"],
        "reporting": ["pdfkit", "Jinja2"],
        "derivations": ["jupyterlab", "sympy"],
        "plotting": ["matplotlib"],
        "dev": ["pytest>=8.3.5", "pyinstrument", "poethepoet"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    ext_modules=build_extensions(),
)
