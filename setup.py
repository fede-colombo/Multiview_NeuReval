# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 09:49:26 2024

@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience, 
         IRCCS San Raffaele Scientific Institute, Milan, Italy
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mvneureval",
    version="1.2.1",
    author="Federica Colombo",
    author_email="fcolombo.italia@gmail.com",
    description="Multiview stability-based relative clustering validation algorithm for multimodal data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fede-colombo/Multiview_NeuReval",
    download_url="https://github.com/fede-colombo/Multiview_NeuReval/releases/tag/v1.2.1",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy",
                      "scipy",
                      "scikit-learn",
                      "umap-learn",
                      "matplotlib",
                      "mvlearn"],
    python_requires='>=3.6',
)
