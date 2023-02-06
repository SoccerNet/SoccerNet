# from distutils.core import setup
import os
import sys
from shutil import rmtree
from setuptools import setup, find_packages, Command

from SoccerNet import __version__, __authors__, __author_email__, __github__

with open('PYPI.md') as readme_file:
    readme = readme_file.read()


class UploadCommand(Command):
    """Support setup.py upload."""

    description = readme + '\n\n',

    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            here = os.path.abspath(os.path.dirname(__file__))
            rmtree(os.path.join(here, 'dist'))
            rmtree(os.path.join(here, 'build'))
            rmtree(os.path.join(here, 'SoccerNet.egg-info'))
        except OSError:
            self.status('Fail to remove previous builds..')
            pass

        # os.system("pyuic5 src/gui/mainwindow.ui -o src/gui/ui_mainwindow.py")

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(
            '{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        # os.system('twine upload --repository-url https://test.pypi.org/legacy/ dist/*')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git commit -am "minor commit for v{0}"'.format(__version__))
        os.system('git push')
        os.system('git tag -d v{0}'.format(__version__))
        os.system('git tag v{0}'.format(__version__))
        os.system('git push --tags')

        sys.exit()


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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    cmdclass={
        'upload': UploadCommand,
    }
)
