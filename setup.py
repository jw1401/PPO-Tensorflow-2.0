from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='RL with TF 2.0',
    version='0.5',
    description='Setup',
    author='jw1401',

    classifiers=[],
    packages=find_packages(),
    zip_safe=False,

    install_requires=[

        'mlagents_envs==0.10',
        'gym',

        'tensorflow',
        'tensorboard',

        'pep8',
        'flake8',
        'pylint',
        'pyaml',
        'matplotlib',
        'cloudpickle',

        'click',
        'flask',
        'flask-cors',
        'websockets',
        'psutil',
        'sqlalchemy'
    ],

    python_requires=">=3.5,<3.8",
)
