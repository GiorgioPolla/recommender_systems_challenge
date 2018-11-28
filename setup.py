import os
from setuptools import setup
setup(
    name = "recommender systems challange",
    version = "0.0.0",
    author = "Giorgio Polla, Paolo Romeo",
    description = ("A set of recommender systems algotirhm"),
    license = "BSD",
    keywords = "example documentation tutorial",
    url = "https://github.com/romeopaolo/recommenderSystems-Polimi2018",
    packages=['RecSys'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
