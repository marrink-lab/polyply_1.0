from collections import OrderedDict
import networkx as nx
import numpy as np
import numpy.linalg
import scipy
import scipy.optimize
from polyply.src.linalg_functions import (angle, dih, u_vect)

def compute_bond(params, coords):
    dist = coords[0] - coords[1]
    return 1000 *(np.linalg.norm(dist) - float(params[1]))**2.0

def compute_angle(params, coords):
    angle_value = angle(coords[0], coords[1], coords[2])
    return (angle_value - float(params[1]))**2.0

def compute_dih(params, coords):
    dih_angle = dih(coords[0], coords[1], coords[2], coords[3])
    return (dih_angle - float(params[1]))**2.0


#def compute_bond(params, coords):
#   dist = coords[0] - coords[1]
#   return 1000 *np.abs(np.linalg.norm(dist) - float(params[1]))

#def compute_angle(params, coords):
#   angle_value = angle(coords[0], coords[1], coords[2])
#   return np.abs(angle_value - float(params[1]))

#def compute_dih(params, coords):
#   dih_angle = dih(coords[0], coords[1], coords[2], coords[3])
#   return np.abs(dih_angle - float(params[1]))



INTER_METHODS = {"bonds": compute_bond,
                 "angles": compute_angle,
                 "dihedrals": compute_dih}

def optimize_geometry(block, coords):
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
