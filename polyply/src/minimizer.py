from collections import OrderedDict
import networkx as nx
import numpy as np
import numpy.linalg
import scipy
import scipy.optimize
from .linalg_functions import (angle, dih, u_vect)

INTER_METHODS = {"bonds": compute_bond,
                 "angles": compute_angle,
                 "dihedrals": compute_dih}

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
    params   list
    coods    numpy array

    Returns
    -------
    float
    """
    dist = np.linalg.norm(coords[0] - coords[1])
    return 1000 *(dist - float(params[1]))**2.0

def compute_angle(params, coords):
    """
    Compute the angle between three points in `coords`
    and then take the MSD with repsect to a reference
    value provided in `params`.

    Parameters
    -----------
    params   list
    coods    numpy array

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
    params   list
    coods    numpy array

    Returns
    -------
    float
    """
    dih_angle = dih(coords[0], coords[1], coords[2], coords[3])
    return (dih_angle - float(params[1]))**2.0


def optimize_geometry(block, coords):
    """
    Take the definitions of a `block` and associated
    `coords` and optimize the geometry based on the
    bonds, angles and dihedrals provided in the
    block definition.

    Parameters
    ----------
    Block  :class:vermouth.molecule.Block
    coords dict
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
        for inter_type in INTER_METHODS:
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

    # optimization failed; we want to relaunch
    if not opt_results['success']:
       return False, coords

    # optimization succeded let's return coordinates
    else:
       positions = opt_results['x'].reshape((-1, 3))
       for node_key, idx in atom_to_idx.items():
           coords[node_key] = positions[idx]
       return True, coords
