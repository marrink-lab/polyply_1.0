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
    return 100*np.abs(dih_angle - float(params[1]))


INTER_METHODS = {"bonds": compute_bond,
                 "angles": compute_angle,
                 "dihedrals": compute_dih}

def callbackF():
    global Nfeval
    print('{0:4d}'.format(Nfeval))
    Nfeval += 1

def optimize_geometry(molecule):
    """
    Simple geometry optimizer for vermuth molecules.
    """
    n_atoms = len(molecule.nodes)
    initial_positions = np.array(list(nx.get_node_attributes(molecule, "position").values()))
    node_to_coord = dict(zip(list(molecule.nodes),range(0, n_atoms)))

    def target_function(positions):
        energy = 0
        positions = positions.reshape((-1, 3))
        for inter_type in molecule.interactions:
            for interaction in molecule.interactions[inter_type]:
                atoms = interaction.atoms
                params = interaction.parameters
                atom_coords = [positions[node_to_coord[name]]
                               for name in atoms]
                if inter_type in ['bonds', 'angles', 'dihedrals']:
                   new = INTER_METHODS[inter_type](params, atom_coords)
                   energy += new

        return energy

    opt_results = scipy.optimize.minimize(target_function, initial_positions, method='L-BFGS-B',
                                          options={'ftol':0.001, 'maxiter': 100})

    print(opt_results)
    positions = opt_results['x'].reshape((-1, 3))

    for name, idx in node_to_coord.items():
        molecule.nodes[name]["positions"] = positions[idx]
