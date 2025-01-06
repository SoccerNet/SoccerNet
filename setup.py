# from distutils.core import setup
import os
import sys
from shutil import rmtree
from setuptools import setup, find_packages, Command

from SoccerNet import __version__, __authors__, __author_email__, __github__

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='SoccerNet',         # How you named your package folder (MyLib)
    packages=['SoccerNet'],   # Chose the same as "name"
    # Start with a small number and increase it with every change you make
    version=__version__,
    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    license='MIT',
    description='SoccerNet SDK',   # Give a short description about your library
    long_description=readme + '\n\n',
    long_description_content_type='text/markdown',
    author=__authors__,                   # Type in your name
    author_email=__author_email__,      # Type in your E-Mail
    url=__github__,   # Provide either the link to your github or to your website
    # download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
    keywords=['SoccerNet', 'SDK', 'Spotting', 'Football', 'Soccer',
              'Video'],   # Keywords that define your package best
    package_data={'SoccerNet': [
        'SoccerNet/data/*.json', 'SoccerNet/data/SNMOT*.txt']},
    include_package_data=True,
    install_requires=[
        'tqdm',
        'scikit-video',
        'matplotlib',
        'google-measurement-protocol',
        'pycocoevalcap',
        # 'sn-trackeval', # broken, trackeval need to fix or install independently
        'huggingface_hub[cli]',
        'boto3',
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        # Define that your audience are developers
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ]
)
