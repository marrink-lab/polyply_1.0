import networkx as nx
import numpy as np
import numpy.linalg
import scipy
import scipy.optimize
import vermouth
import polyply
from .minimizer import optimize_geometry
from .processor import Processor
from .linalg_functions import (angle, dih, u_vect, center_of_geometry,
                                          norm_sphere, radius_of_gyration)
from .random_walk import _take_step
from .topology import replace_defined_interaction
from .virtual_site_builder import construct_vs
"""
Processor generating coordinates for all residues
of a meta_molecule matching those in the meta_molecule.molecule attribute.
"""

def find_atoms(molecule, attr, value):
    """
    Find all nodes of a `vermouth.molecule.Molecule` that have the
    attribute `attr` with the corresponding value of `value`.

    Parameters
    ----------
    molecule: :class:vermouth.molecule.Molecule
    attr: str
         attribute that a node needs to have
    value:
         corresponding value

    Returns
    ----------
    list
       list of nodes found
    """
    nodes=[]
    for node in molecule.nodes:
        if molecule.nodes[node][attr] == value and attr in molecule.nodes[node]:
              nodes.append(node)

    return nodes

def find_step_length(interactions, current_node, prev_node):
    """
    Given a list of `interactions` in vermouth format, find an
    interaction from bonds, constraints, or virtual-sites that
    involves the `current_node` and `prev_node`. Return if this
    interaction is a virtual-site and the corresponding parameter.

    Parameters
    -----------
    interactions:  :class:dict
         interaction dictionary
    current_node:   int
         node index
    prev_node:      int
         node index

    Returns:
    ---------
    bool
      is the interaction a virtual-site
    :tuple:vermouth.interaction
    """
    for inter_type in ["bonds", "constraints", "virtual_sitesn",
                       "virtual_sites2", "virtual_sites3", "virtual_sites4" ]:
        inters = interactions.get(inter_type, [])
        for interaction in inters:
            if current_node in interaction.atoms:
               if prev_node in interaction.atoms and inter_type in ["bonds", "constraints"]:
                  return False, interaction, inter_type
               elif prev_node in interaction.atoms and inter_type.split("_")[0] == "virtual":
                  return True, interaction, inter_type

def _expand_inital_coords(block):
    """
    Given a `block` generate initial random coordinates
    for all atoms in the block. Note that the initial
    coordinates though random, have a defined distance
    correspodning to a bond , constaint or virtual-site.

    Parameters
    -----------
    block:   :class:`vermouth.molecule.Block`

    Returns
    ---------
    dict
      dictonary of node index and position
    """
    coords = {}
    #TODO this should actually be the index
    atom = list(block.nodes)[0]
    coords[atom] = np.array([0, 0, 0])

    vectors = norm_sphere(values=1000)
    for prev_node, current_node in nx.dfs_edges(block, source=atom):
        prev_coord = coords[prev_node]
        is_vs, interaction, inter_type = find_step_length(block.interactions,
                                                          current_node,
                                                          prev_node)
        if is_vs:
            coords[current_node] = construct_vs(inter_type, interaction, coords)
        else:
            coords[current_node], _ = _take_step(vectors,
                                                 float(interaction.parameters[1]),
                                                 prev_coord)
    return coords

def compute_volume(molecule, block, coords):
    """
    Given a `block`, which is part of `molecule` and
    has the coordinates `coord` compute the radius
    of gyration taking into account the volume of each
    particle. The volume of a particle is considered to be
    the sigma value of it's LJ self interaction parameter.

    Parameters
    ----------
    molecule:  :class:vermouth.molecule.Molecule
    block:     :class:vermouth.molecule.Block
    coords:    :class:dict
      dictionary of positions in from node_idx: np.array

    Returns
    -------
    radius of gyration
    """
    n_atoms = len(coords)
    points = np.array(list(coords.values()))
    CoG = center_of_geometry(points)
    geom_vects = np.zeros((n_atoms, 3))
    idx = 0

    for node, coord in coords.items():
        atom_key = block.nodes[node]["atype"]
        rad = float(molecule.nonbond_params[frozenset([atom_key, atom_key])]["nb1"])
        diff = coord - CoG
        geom_vects[idx, :] = diff + u_vect(diff) * rad
        idx += 1

    if geom_vects.shape[0] > 1:
        radgyr = radius_of_gyration(geom_vects)
    else:
        radgyr = rad
    return radgyr

def map_from_CoG(coords):
    """
    Compute the center of geometry
    of `coords` and return each position
    as vector between the center of geometry
    and the original positon.

    Parameters
    ----------
    coords:   :class:dict
        dictionary of coordinates

    Returns
    --------
    dict
     dictionary of node idx and CoG vector
    """
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

    return True

def extract_block(molecule, resname, defines):
    """
    Given a `vermouth.molecule` and a `resname`
    extract the information of a block form the
    molecule definition and replace all defines
    if any are found.

    Parameters
    ----------
    molecule:  :class:vermouth.molecule.Molecule
    resname:   str
    defines:   dict
      dict of type define:value

    Returns
    -------
    :class:vermouth.molecule.Block
    """
    nodes = find_atoms(molecule, "resname", resname)
    resid = molecule.nodes[nodes[0]]["resid"]
    block = vermouth.molecule.Block()

    for node in nodes:
        attr_dict = molecule.nodes[node]
        if attr_dict["resid"] == resid:
            block.add_node(node, **attr_dict)

    for inter_type in molecule.interactions:
        for interaction in molecule.interactions[inter_type]:
            if all(atom in block for atom in interaction.atoms):
               interaction = replace_defined_interaction(interaction, defines)
               block.interactions[inter_type].append(interaction)

    for inter_type in ["bonds", "constraints", "virtual_sitesn"]:
        block.make_edges_from_interaction_type(inter_type)

    return block

class GenerateTemplates(Processor):
    """
    This processor takes a a class:`polyply.src.MetaMolecule` and
    creates a block for each unique residue type in the molecule
    as well as positions for that block. These blocks are stored
    in the templates attribute. The processor also stores the volume
    of each block in the volume attribute.
    """

    def _gen_templates(self, meta_molecule):
        """
        Generate blocks for each unique residue by extracting the
        block information, placining inital cooridnates, and geometry
        optimizing those coordinates. Subsquently compute volume.

        Parameters
        ----------
        meta_molecule: :class:`polyply.src.meta_molecule.MetaMolecule`

        Returns
        ---------
        templates  dict
           dict of resname:block
        volumes    dict
           dict of name:volume
        """
        resnames = set(nx.get_node_attributes(meta_molecule.molecule,
                                              "resname").values())
        templates = {}
        volumes = {}

        for resname in resnames:
            block = extract_block(meta_molecule.molecule, resname,
                                  meta_molecule.defines)
            opt_counter=0
            while True:
                coords = _expand_inital_coords(block)
                success, coords = optimize_geometry(block, coords)
                if success:
                   break
                elif opt_counter > 10:
                   print("Warning: Failed to optimize structure for block {}.".format(resname))
                   print("Proceeding with unoptimized coordinates.")
                   break
                else:
                   opt_counter += 1

            volumes[resname] = compute_volume(meta_molecule, block, coords)
            coords = map_from_CoG(coords)
            templates[resname] = coords

        return templates, volumes

    def run_molecule(self, meta_molecule):
        """
        Execute the generation of templates and set the template
        and volume attribute.
        """
        templates, volumes = self._gen_templates(meta_molecule)
        meta_molecule.templates = templates
        meta_molecule.volumes = volumes
        return meta_molecule
