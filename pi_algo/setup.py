from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.md')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='pi_algo',
    version='0.0.0',
    description=('Computing metrics associated to step by step protocol.'),
    long_description=long_description,
    author='Step by Step project',
    author_email='undefined@gmail.com',
    url='https://github.com/',
    license='undefined',
    packages=['pi_algo'],
    package_data={'pi_algo': ['tests/input/models/*.sav']},
    scripts=['script/run_protocol_1'],
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6'],
    )
