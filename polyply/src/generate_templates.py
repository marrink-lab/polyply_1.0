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

def construct_vs(atoms, coords):
   coord = np.array([0., 0., 0.])
   for atom in atoms[1:]:
       if atom in coords:
          coord += coords[atom]
   return coord



def find_step_length(interactions, current_node, prev_node):
    for inter_type in interactions:
        if inter_type in ["bonds", "constraints", "virtual_sitesn"]:
           for interaction in interactions[inter_type]:
               if current_node in interaction.atoms:
                  if prev_node in interaction.atoms and inter_type != "virtual_sitesn":
                     return False, float(interaction.parameters[1])
                  elif inter_type == "virtual_sitesn":
                     return True, interaction.atoms

def _expand_inital_coords(block, coords):

   if not coords:
      atom = list(block.nodes)[0]
      coords[atom] = np.array([0, 0, 0])

   vectors = norm_sphere(values=1000)
   for prev_node, current_node in nx.dfs_edges(block, source=atom):
       prev_coord = coords[prev_node]
       is_vs, param = find_step_length(block.interactions, current_node, prev_node)
       if is_vs:
           coords[current_node] = construct_vs(atoms, coords)
       else:
          coords[current_node], _ = _take_step(vectors, param, prev_coord)

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
           if A == 0 and B == 0 and atom_key != "H":
              A = float(molecule.nonbond_params[(atom_key, atom_key)]["nb1"])
              B = float(molecule.nonbond_params[(atom_key, atom_key)]["nb2"])
              rad = 1.22*(A/B)**(1/6.)
           else:
              rad = 0

        else:
           rad = 1.22*float(molecule.atom_types[atom_key]["nb1"])
           if rad == 0 and atom_key != "H":
              rad = 1.22*float(molecule.nonbond_params[(atom_key, atom_key)]["nb1"])
           else:
              rad =0

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

def replace_defines(interaction, defines):
    def_key = interaction.parameters[-1]
    if def_key in defines:
       values = defines[def_key]
       del interaction.parameters[-1]
       [interaction.parameters.append(param) for param in values]

    return interaction

def extract_block(molecule, resname, defines):

    nodes = find_atoms(molecule, "resname", resname)
    resid = molecule.nodes[nodes[0]]["resid"]
    block = vermouth.molecule.Block()

    for node in nodes:
        attr_dict = molecule.nodes[node]
        if attr_dict["resid"] == resid:
           block.add_node(node, **attr_dict)

    for inter_type in molecule.interactions:
        for interaction in molecule.interactions[inter_type]:
            if _atoms_in_node(interaction.atoms, block.nodes):
               interaction = replace_defines(interaction, defines)
               block.interactions[inter_type].append(interaction)

    for inter_type in ["bonds", "constraints"]:
        block.make_edges_from_interaction_type(inter_type)

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
            block = extract_block(meta_molecule.molecule, resname, 
                                  meta_molecule.defines)
            coords = _expand_inital_coords(block, {})
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
