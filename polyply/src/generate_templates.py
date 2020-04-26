import networkx as nx
import numpy as np
import numpy.linalg
import scipy
import scipy.optimize
import vermouth
import polyply
from polyply.src.minimizer import optimize_geometry
from polyply.src.processor import Processor
from polyply.src.linalg_functions import (angle, dih, u_vect, center_of_geometry,
                                          norm_sphere, radius_of_gyration)
from polyply.src.random_walk import _take_step

def find_atoms(molecule, attr, value):
    nodes=[]
    for node in molecule.nodes:
        if attr in molecule.nodes[node]:
           if molecule.nodes[node][attr] == value:
              nodes.append(node)

    return nodes

def _expand_inital_coords(block, coords, inter_type):

    if not coords:
       atom = list(block.nodes)[0]
       coords[atom] = np.array([0, 0, 0])

    vectors = norm_sphere(values=1000)
    for bond in block.interactions[inter_type]:
        atoms = bond.atoms
        params = bond.parameters
        if atoms[0] in coords and atoms[1] not in coords:
            dist = float(params[1])
            coords[atoms[1]], _ = _take_step(vectors, dist ,coords[atoms[0]])

        elif atoms[1] in coords and atoms[0] not in coords:
            dist = float(params[1])
            coords[atoms[0]], _ = _take_step(vectors, dist, coords[atoms[1]])

        else:
            continue

    return coords


def compute_volume(molecule, block, coords):
    n_atoms = len(coords)
    points = np.array(list(coords.values()))
    CoG = center_of_geometry(points)
    geom_vects = np.zeros((n_atoms, 3))
    idx = 0

    for node, coord in coords.items():
        atom_key = block.nodes[node]["atype"]

        if molecule.defaults["nbfunc"] == 1:
           A = float(molecule.atom_types[atom_key]["nb1"])
           B = float(molecule.atom_types[atom_key]["nb2"])
           rad = 1.22*(A/B)**(1/6.)
        else:
           rad = 1.22*float(molecule.atom_types[atom_key]["nb1"])

        diff = coord - CoG
        geom_vects[idx, :] = diff + u_vect(diff) * rad
        idx += 1

    if geom_vects.shape[0] > 1:
       radgyr = radius_of_gyration(geom_vects)
    else:
       radgyr = rad
    return radgyr

def map_from_CoG(coords):
    n_atoms = len(coords)
    points = np.array(list(coords.values()))
    CoG = center_of_geometry(points)
    out_vectors = {}
    for key, coord in coords.items():
        diff = coord - CoG
        out_vectors[key] = diff

    return out_vectors

def _atoms_in_node(atoms, nodes):
    for atom in atoms:
        if atom not in nodes:
           return False
    else:
        return True

def extract_block(molecule, resname):

    nodes = find_atoms(molecule, "resname", resname)
    resid = molecule.nodes[nodes[0]]["resid"]
    block = vermouth.molecule.Molecule()

    for node in nodes:
        attr_dict = molecule.nodes[node]
        if attr_dict["resid"] == resid:
           block.add_node(node, **attr_dict)

    for inter_type in molecule.interactions:
        for interaction in molecule.interactions[inter_type]:
            if _atoms_in_node(interaction.atoms, block.nodes):
               block.interactions[inter_type].append(interaction)

    return block

class GenerateTemplates(Processor):
    """

    """

    def _gen_templates(self, meta_molecule):
        resnames = set(nx.get_node_attributes(meta_molecule.molecule,
                                              "resname").values())
        templates = {}
        volumes = {}

        for resname in resnames:
            block = extract_block(meta_molecule.molecule, resname)
            coords = _expand_inital_coords(block, {}, 'bonds')
            coords = _expand_inital_coords(block, coords, 'constraints')
            coords = optimize_geometry(block, coords)
            volumes[resname] = compute_volume(meta_molecule, block, coords)
            coords = map_from_CoG(coords)
            templates[resname] = coords

        return templates, volumes

    def run_molecule(self, meta_molecule):
        templates, volumes = self._gen_templates(meta_molecule)
        meta_molecule.templates = templates
        meta_molecule.volumes = volumes
        return meta_molecule
