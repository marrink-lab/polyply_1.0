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

"""
Parser for coordinate files, which allows for the backmapping of
a system on a user provided configuration.
"""

import numpy as np

def read_xyz(path):
    """
    Read an xyz file and extract molecule's coordinates.

    The file should be in the format::

        <number of atoms>
        comment line
        <element0> <X0> <Y0> <Z0>
        <element1> <X1> <Y1> <Z1>
        ...

    Parameters
    ----------
    path : str
        Path to the xyz file to read

    Returns
    -------
    coords : np.ndarray
        array of atom coordinates with shape (len(molecule.nodes), 3)
    """

    coords = []
    with open(path, 'r') as xyz_file:
        num_atoms = int(xyz_file.readline())
        next(xyz_file)

        for line in xyz_file:
            _, x, y, z = line.strip().split()
            coords.append([float(x), float(y), float(z)])

        if num_atoms != len(coords):
            raise IOError('<number of atoms> not equal to number'
                          'of coordinates in xyz file'
                          )
    return np.array(coords)
