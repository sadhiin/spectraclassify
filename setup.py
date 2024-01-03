from setuptools import setup, find_packages
import subprocess

with open("README.md", "r") as fh:
    desc = fh.read()

__version__ = "0.0.1"

REPO_NAME = "SpectraClassify"
AUTHOOR_NAME = "Sadhin"
SRC_REPO = "spectraclassify"
AUTHOR_EMAIL = "sadhin.aiub.cse@gmail.com"


setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOOR_NAME,
    author_email=AUTHOR_EMAIL,
    description="SpectraClassify is a python package for zero code image classification within the browser",
    long_description = desc,
    long_description_content_type="text/markdown",
    url=f"",
    install_requires=[
        'tensorflow==2.13',
        'numpy',
        'Pillow',
        'Flask',
        'Flask-Cors'
    ],
    classifiers=[
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

    # Pick your license as you wish
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.10',
  ],
  entry_points={
        "console_scripts": [
            "run = spectraclassify.main:runnapplication",
        ]},
)
