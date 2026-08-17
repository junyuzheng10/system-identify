from setuptools import setup, find_packages
from os import path

__version__ = "0.0.1"
here = path.abspath(path.dirname(__file__))

requires_list = []
#with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
#    for line in f:
#        requires_list.append(str(line))

long_description = 'system_identification is a Python library used to do system identification.'

setup(
    name='system-identification',
    version=__version__,
    description='A Python toolkit for system identification.',
    long_description=long_description,
    author="zhengjunyu",
    license='MIT',
    packages=[package for package in find_packages()
              if package.startswith('system_identification')],
    zip_safe=False,
    install_requires=requires_list,
    extras_require={},
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 ]
)
