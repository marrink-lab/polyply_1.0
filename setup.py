# Copyright 2020 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

setup(
    name='polyply',
    version='0.1.1',
    author='Fabian Grünewald',
    author_email='f.grunewald@rug.nl',
    packages=find_packages(),
    include_package_data=True,
    scripts=['bin/polyply', ],
    license='Apache 2.0',
    description='tool for generating GROMACS (bio)-macromolecule itps and structures',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'scipy','networkx','tqdm', 'vermouth'],
)
