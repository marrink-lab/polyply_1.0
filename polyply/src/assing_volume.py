import networkx as nx
import numpy as np
import numpy.linalg
import scipy
import scipy.optimize
import polyply
from polyply.src.processor import Processor
from polyply.src.geometrical_functions import (angle, dih, u_vect)

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
                atom_coords = [positions[atom_to_idx[name]]
                               for name in atoms]
                if inter_type in ['bonds', 'angles', 'dihedrals']:
                   new = INTER_METHODS[inter_type](params, atom_coords)
                   energy += new
        return energy

    opt_results = scipy.optimize.minimize(target_function, positions, method='L-BFGS-B',
                                          options={'ftol':0.001, 'maxiter': 100})

    print(opt_results)
    positions = opt_results['x'].reshape((-1, 3))

    for name, idx in atom_to_idx.items():
        coords[name] = positions[idx]

    return coords


def _take_random_step(ref, step_length, values=50):
    vectors = polyply.src.geometrical_functions.norm_sphere(values=values)
    index = np.random.randint(0, len(vectors) - 1)
    new_point = ref + vectors[index] * step_length
    return new_point

def _expand_inital_coords(block, coords, inter_type):

    for bond in block.interactions[inter_type]:
        atoms = bond.atoms
        params = bond.parameters
        if not coords:
            coords[atoms[0]] = np.array([0, 0, 0])
            dist = float(params[1])
            coords[atoms[1]] = np.array([0, 0, dist])
        else:
            if atoms[0] in coords and atoms[1] not in coords:
                dist = float(params[1])
                coords[atoms[1]] = _take_random_step(coords[atoms[0]], dist)

            elif atoms[1] in coords and atoms[0] not in coords:
                dist = float(params[1])
                coords[atoms[0]] = _take_random_step(coords[atoms[1]], dist)

            else:
                continue

    return coords

def radius_of_gyration(traj):
    N = len(traj)
    diff=np.zeros((N**2))
    count=0
    for i in traj:
        for j in traj:
            diff[count]=np.dot((i - j),(i-j))
            count = count + 1
    Rg= 1/np.float(N)**2 * sum(diff)
    return(np.float(np.sqrt(Rg)))

def center_of_geomtry(points):
    return np.average(points, axis=0)

def compute_volume(block, coords, vdwradii):
    n_atoms = len(coords)
    points = np.array(list(coords.values()))
    CoG = center_of_geomtry(points)
    geom_vects = np.zeros((n_atoms, 3))
    idx = 0
    for atom_key, coord in coords.items():
        atom = block.nodes[atom_key]["atomname"]
        diff = coord - CoG
        rad = vdwradii[atom]
        geom_vects[idx, :] = diff + u_vect(diff) * rad
        idx += 1

    radgyr = radius_of_gyration(geom_vects)
    return radgyr


class GenerateTemplates(Processor):
    """

    """

    def _gen_templates(self, meta_molecule):
        resnames = set(nx.get_node_attributes(meta_molecule.molecule,
                                              "resname").values())
        templates = {}
        vdwradii = {'C1': 0.17,
                    'C2'  : 0.17,
                    'C3': 0.17,
                    'C4'  : 0.17,
                    'O1'  : 0.152,
                    'O2' : 0.152,
                    'C5': 0.17  }
        volumes = {}
        for resname in resnames:
            block = meta_molecule.force_field.blocks[resname]
            coords = _expand_inital_coords(block, {}, 'bonds')
            coords = _expand_inital_coords(block, coords, 'constraints')
            coords = energy_minimize(block, coords)
            templates[resname] = coords
            volumes[resname] = compute_volume(block, coords, vdwradii)
           # with open(resname + ".xyz", 'w') as _file:
           #      _file.write("{} \n \n".format(len(coords)))
           #      for name, xyz in coords.items():
           #          _file.write('{} {} {} {}\n'.format(name, 10*xyz[0], 10*xyz[1],10* xyz[2]))

        return templates, volumes

    def run_molecule(self, meta_molecule):
        templates, volumes = self._gen_templates(meta_molecule)
        meta_molecule.templates = templates
        meta_molecule.volumes = volumes
        return meta_molecule
