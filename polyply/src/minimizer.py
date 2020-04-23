import networkx as nx
import numpy as np
import numpy.linalg
import scipy
import scipy.optimize
from polyply.src.linalg_functions import (angle, dih, u_vect)

def compute_bond(params, coords):
    dist = coords[0] - coords[1]
    return 1000* np.abs(np.linalg.norm(dist) - float(params[1]))

def compute_angle(params, coords):
    angle_value = angle(coords[0], coords[1], coords[2])
    return np.abs(angle_value - float(params[1]))

def compute_dih(params, coords):
    dih_angle = dih(coords[0], coords[1], coords[2], coords[3])
    return np.abs(dih_angle - float(params[1]))


INTER_METHODS = {"bonds": compute_bond,
                 "angles": compute_angle,
                 "dihedrals": compute_dih}

def energy_minimize(block, coords):
    n_atoms = len(coords)
    atom_to_idx = dict(zip(list(coords.keys()),range(0, n_atoms)))
    positions = np.array(list(coords.values()))
    def target_function(positions):
        energy = 0
        positions = positions.reshape((-1, 3))
        for inter_type in block.interactions:
            for interaction in block.interactions[inter_type]:
                atoms = interaction.atoms
                params = interaction.parameters
                atom_coords = [positions[name - 1 ]
                               for name in atoms]
                if inter_type in ['bonds', 'angles', 'dihedrals']:
                   new = INTER_METHODS[inter_type](params, atom_coords)
                   energy += new
        return energy

    opt_results = scipy.optimize.minimize(target_function, positions, method='L-BFGS-B',
                                          options={'ftol':0.001, 'maxiter': 100})

    #print(opt_results)
    positions = opt_results['x'].reshape((-1, 3))
    #print(coords)
    #print(positions)
    for idx in coords:
        coords[idx] = positions[idx-1]

    return coords
