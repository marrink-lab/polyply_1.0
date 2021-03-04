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

from collections import OrderedDict
import numpy as np
import scipy
import scipy.optimize
from .linalg_functions import (angle, dih)
from .virtual_site_builder import construct_vs
"""
Processor and functions for optimizing the geomtry
of vermoth molecules.
"""

def compute_bond(params, coords):
    """
    Compute the distance between two points in `coords`
    and then take the MSD with repsect to a reference
    value provided in `params` multiplyed by 1000. The
    factor 1000 enforces that the bounds get a higher
    weight in optimization than angles and dihedrals,
    as we deal in units of nm.

    Parameters
    -----------
    params:   list
    coodrs:    numpy.ndarray

    Returns
    -------
    float
    """
    dist = np.linalg.norm(coords[0] - coords[1])
    return 10000 *(dist - float(params[1]))**2.0

def compute_angle(params, coords):
    """
    Compute the angle between three points in `coords`
    and then take the MSD with respect to a reference
    value provided in `params`.

    Parameters
    -----------
    params:   list
    coords:    numpy.ndarray

    Returns
    -------
    float
    """
    angle_value = angle(coords[0], coords[1], coords[2])
    return (angle_value - float(params[1]))**2.0

def compute_dih(params, coords):
    """
    Compute the dihedral angle between four points in `coords`
    and then take the MSD with repsect to a reference
    value provided in `params`.

    Parameters
    -----------
    params:   list
    coords:    numpy.ndarray

    Returns
    -------
    float
    """
    # only optimize imporper dihedrals
    # we filter all non-imporpers out here because
    # the itp file parser doesn't distinguish them
    if params[0] == "2":
        dih_angle = dih(coords[0], coords[1], coords[2], coords[3])
        penalty = (dih_angle - float(params[1]))**2.0
    else:
        penalty = 0
    return penalty

def renew_vs(positions, block, atom_to_idx):
    """
    Given `positions` update the virtual-sites
    that are specified in block using the current
    positions of the other atoms.

    Parameters
    ----------
    positions: dict
    block: :class:vermouth.molecule.Block

    Returns
    -------
    positions
      the positions with new vs coordinates
    """
    vs_types = ["virtual_sitesn", "virtual_sites2", "virtual_sites3", "virtual_sites4"]
    for vs_type in vs_types:
        interactions = block.interactions.get(vs_type, [])
        for virtual_site in interactions:
            vs_tb = atom_to_idx[virtual_site.atoms[0]]
            pos_dict = {}
            for atom in virtual_site.atoms:
                pos_dict[atom] = positions[atom_to_idx[atom]]
            new_vs = construct_vs(vs_type, virtual_site, pos_dict)
            positions[vs_tb] = new_vs
    return positions


INTER_METHODS = {"bonds": compute_bond,
                 "constraints": compute_bond,
                 "angles": compute_angle,
                 "dihedrals": compute_dih}

def optimize_geometry(block, coords, inter_types=[]):
    """
    Take the definitions of a `block` and associated
    `coords` and optimize the geometry based on the
    bonds, angles and dihedrals provided in the
    block definition.

    Parameters
    ----------
    block:  :class:vermouth.molecule.Block
    coords: dict
        dictionary of coordinates in form atom_name:np.ndarray

    Returns
    -------
    bool
      status of the optimization i.e. failure or success
    dict
      dictionary of optimized coordinates
    """
    n_atoms = len(coords)
    atom_to_idx = OrderedDict(zip(list(coords.keys()), range(0, n_atoms)))
    positions = np.array(list(coords.values()))

    def target_function(positions):
        energy = 0
        positions = positions.reshape((-1, 3))
        positions = renew_vs(positions, block, atom_to_idx)
        for inter_type in inter_types:
            interactions = block.interactions.get(inter_type, [])
            for interaction in interactions:
                atoms = interaction.atoms
                params = interaction.parameters
                atom_coords = [positions[atom_to_idx[name]]
                               for name in atoms]
                new = INTER_METHODS[inter_type](params, atom_coords)
                energy += new
        return energy

    opt_results = scipy.optimize.minimize(target_function, positions, method='L-BFGS-B',
                                          options={'ftol':0.001, 'maxiter': 100})

    positions = opt_results['x'].reshape((-1, 3))
    positions = renew_vs(positions, block, atom_to_idx)
    for node_key, idx in atom_to_idx.items():
        coords[node_key] = positions[idx]

    return opt_results['success'], coords
