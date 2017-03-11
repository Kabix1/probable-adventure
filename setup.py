import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Chess Stream Analyser",
    version = "0.0.1",
    author = "Olle Wiklund and Emil Henning Bruce",
    author_email = "owl@live.se, emil.henning@gmail.com",
    description = ("Gets chess positions from a stream of a chess game"
                                   ""),
    license = "BSD",
    url = "https://github.com/Kabix1/probable-adventure",
    packages=['ChessVideoAnalyser', 'Othello'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=['tensorflow', 'imageio', 'pafy', 'pylab'],
)
