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
from itertools import islice

def read_dat(path):
    """
    Read a configuration (.dat) file and extract molecule coordinates
    and rotation. The Backbone-base vector and Normal vector are optional,
    if not assigned they will be generated during the backmapping.
    Additionally, only the normal can be provided and the corresponding
    reference frame will be constructed.

    R: Position vector
    N: Normal vector
    T: Tangent/backbone-base vector

    The file should be in the format::

        <number of atoms>
        comment line
        <Rx> <Ry> <Rz> <Nx> <Ny> <Nz> <Tx> <Ty> <Tz>
        ...

    Parameters
    ----------
    path : str
        Path to the configuration (.dat) file to read

    Returns
    -------
    positions : np.ndarray
        array of position vectors with shape (len(molecule.nodes), 3)
    normals : np.ndarray
        array of normal vectors with shape (len(molecule.nodes), 3)
    tangents : np.ndarray
        array of tangent/backbone-base vectors with shape (len(molecule.nodes), 3)
    """

    positions, normals, tangents = [], [], []
    with open(path, 'r') as dat_file:
        num_atoms = int(dat_file.readline())

        for line in islice(dat_file, 1, None):
            values = [float(element) for element in line.split()]
            position, normal, tangent = values[:3], values[3:6], values[6:9]
            positions.append(position), normals.append(normal), tangents.append(tangent)

        if num_atoms != len(positions):
            raise IOError('<number of atoms> not equal to number'
                          'of coordinates in dat file'
                          )
    return np.asarray(positions), np.asarray(normals), np.asarray(tangents)
