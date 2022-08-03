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
import networkx as nx
import numpy as np
import vermouth
from vermouth.log_helpers import StyleAdapter, get_logger
from .minimizer import optimize_geometry
from .processor import Processor
from .linalg_functions import (u_vect, center_of_geometry,
                               radius_of_gyration)
from .topology import replace_defined_interaction
from .linalg_functions import dih
"""
Processor generating coordinates for all residues of a meta_molecule
matching those in the meta_molecule.molecule attribute.
"""

LOGGER = StyleAdapter(get_logger(__name__))

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
    nodes = []
    for node in molecule.nodes:
        if attr in molecule.nodes[node] and molecule.nodes[node][attr] == value:
            nodes.append(node)

    return nodes

def find_interaction_involving(block, current_node, prev_node):
    """
    Given a list of `interactions` in vermouth format, find an
    interaction from bonds, constraints, or virtual-sites that
    involves the `current_node` and `prev_node`. Return if this
    interaction is a virtual-site and the corresponding parameter.

    Parameters
    -----------
    block:  :class:`vermouth.molecule.Block`
         vermouth block
    current_node:   int
         node index
    prev_node:      int
         node index

    Returns:
    ---------
    bool
      is the interaction a virtual-site
    vermouth.Interaction
      interaction definition
    str
      interaction type
    """
    interactions = block.interactions
    for inter_type in ["bonds", "constraints", "virtual_sitesn",
                       "virtual_sites2", "virtual_sites3", "virtual_sites4"]:
        inters = interactions.get(inter_type, [])
        for interaction in inters:
            if current_node in interaction.atoms:
                if prev_node in interaction.atoms and inter_type in ["bonds", "constraints"]:
                    return False, interaction, inter_type
                elif prev_node in interaction.atoms and inter_type.split("_")[0] == "virtual":
                    return True, interaction, inter_type
    msg = '''Cannot build template for residue {}. No constraint, bond, virtual-site
             linking atom {} and atom {}.'''
    raise IOError(msg.format(block.nodes[0]["resname"], prev_node, current_node))

def _expand_inital_coords(block, max_count=50000):
    """
    Given a `graph` generate initial coordinates in three dimensions
    using the Kamada-Kawai algorithm.

    Parameters
    -----------
    block:   :class:`vermouth.molecule.Block`
    max_count: int
        maximum number of iterations trying to fix the improper dihedrals

    Returns
    ---------
    dict
      dictonary of node index and position
    """
    count = 0
    while True:
        coords = nx.kamada_kawai_layout(block, dim=3)
        count += 1
        if count > max_count or _good_impropers(coords, block):
            break

    return coords

def _good_impropers(coords, block):
    """
    Given the coords of the block, check if the sign of the improper
    dihedrals defined in the interaction section is satisfied by the
    coordinates. This ensures that coordinates are in an initial
    geometry which can be optimzied later.

    Parameters
    ----------
    coords: dict
    block: :class:`vermouth.molecule.Block`

    Returns
    -------
    bool
    """
    for improper in block.interactions["dihedrals"]:
        if improper.parameters[0] == "2":
            atoms = improper.atoms
            dih_angle = dih(coords[atoms[0]], coords[atoms[1]], coords[atoms[2]], coords[atoms[3]])
            if np.isclose(np.abs(float(improper.parameters[1])), 0):
                continue

            if np.sign(dih_angle) != np.sign(float(improper.parameters[1])):
                return False
    return True

def compute_volume(block, coords, nonbond_params, treshold=1e-18):
    """
    Given a `block`, which has the coordinates `coords`,
    compute the radius of gyration taking into account
    the volume of each particle. The volume of a particle
    is considered to be the sigma value of it's LJ self-
    interaction parameter.

    Parameters
    ----------
    block:     :class:`vermouth.molecule.Block`
    coords:    dict[abc.hashable]
        dictionary of positions in from node_idx: np.array
    nonbond_params: dict[frozenset(abc.hashable, abc.hashable)]
        dictionary of nonbonded parameters with atom-types as
        keys and sigma, epsilon LJ parameters
    treshold: float
        distance from center of geometry at which the
        particle is not taken into account for the volume
        computation

    Returns
    -------
    radius of gyration
    """
    n_atoms = len(coords)
    points = np.array(list(coords.values()))
    res_center_of_geometry = center_of_geometry(points)
    geom_vects = np.zeros((n_atoms, 3))
    idx = 0

    for node, coord in coords.items():
        atom_key = block.nodes[node]["atype"]
        rad = float(nonbond_params[frozenset([atom_key, atom_key])]["nb1"])
        diff = coord - res_center_of_geometry
        if np.linalg.norm(diff) > treshold:
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
    points = np.array(list(coords.values()))
    res_center_of_geometry = center_of_geometry(points)
    out_vectors = {}
    for key, coord in coords.items():
        diff = coord - res_center_of_geometry
        out_vectors[key] = diff

    return out_vectors

def _relabel_interaction_atoms(interaction, mapping):
    """
    Relables the atoms in interaction according to the
    rules defined in mapping.

    Parameters
    ----------
    interaction: `vermouth.molecule.Interaction`
    mapping: `:class:dict`

    Returns
    -------
    interaction: `vermouth.molecule.Interaction`
        the new interaction with updated atoms
    """
    new_atoms = [mapping[atom] for atom in interaction.atoms]
    new_interaction = interaction._replace(atoms=new_atoms)
    return new_interaction

def extract_block(molecule, resname, defines):
    """
    Given a `vermouth.molecule` and a `resname`
    extract the information of a block from the
    molecule definition and replace all defines
    if any are found.

    Parameters
    ----------
    molecule:  :class:vermouth.molecule.Molecule
    resname:   str
    defines:   dict
      dict of type define: value

    Returns
    -------
    :class:vermouth.molecule.Block
    """
    nodes = find_atoms(molecule, "resname", resname)
    resid = molecule.nodes[nodes[0]]["resid"]
    block = vermouth.molecule.Block()

    # select all nodes with the same first resid and
    # make sure the block node labels are atomnames
    # also build a correspondance dict between node
    # label in the molecule and in the block for
    # relabeling the interactions
    mapping = {}
    for node in nodes:
        attr_dict = molecule.nodes[node]
        if attr_dict["resid"] == resid:
            block.add_node(attr_dict["atomname"], **attr_dict)
            mapping[node] = attr_dict["atomname"]

    for inter_type in molecule.interactions:
        for interaction in molecule.interactions[inter_type]:
            if all(atom in mapping for atom in interaction.atoms):
                interaction = replace_defined_interaction(interaction, defines)
                interaction = _relabel_interaction_atoms(interaction, mapping)
                block.interactions[inter_type].append(interaction)

    for inter_type in ["bonds", "constraints", "virtual_sitesn",
                       "virtual_sites2", "virtual_sites3", "virtual_sites4"]:
        block.make_edges_from_interaction_type(inter_type)

    if not nx.is_connected(block):
        msg = ('\n Residue {} with id {} consistes of two disconnected parts. '
               'Make sure all atoms/particles in a residue are connected by bonds,'
               ' constraints or virual-sites.')
        raise IOError(msg.format(resname, resid))

    return block

class GenerateTemplates(Processor):
    """
    This processor takes a a class:`polyply.src.MetaMolecule` and
    creates a block for each unique residue type in the molecule
    as well as positions for that block. These blocks are stored
    in the templates attribute. The processor also stores the volume
    of each block in the volume attribute.
    """
    def __init__(self, topology, max_opt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_opt = max_opt
        self.topology = topology
        self.volumes = self.topology.volumes
        self.templates = {}

    def gen_templates(self, meta_molecule):
        """
        Generate blocks for each unique residue by extracting the
        block information, placing initial coordinates, and geometry
        optimizing those coordinates. Subsequently compute the volume
        for each block unless they are already given in the volumes
        class variable.

        Parameters
        ----------
        meta_molecule: :class:`polyply.src.meta_molecule.MetaMolecule`

        Updates
        ---------
        self.templates
        self.volumes
        """
        resnames = set(nx.get_node_attributes(meta_molecule, "resname").values())

        for resname in resnames:
            if resname not in self.templates:
                block = extract_block(meta_molecule.molecule, resname,
                                      self.topology.defines)

                opt_counter = 0
                while True:

                    coords = _expand_inital_coords(block)
                    success, coords = optimize_geometry(block,
                                                        coords,
                                                        ["bonds", "constraints", "angles"])

                    success, coords = optimize_geometry(block,
                                                        coords,
                                                        ["bonds", "constraints", "angles", "dihedrals"])

                    if success:
                        break
                    elif opt_counter > self.max_opt:
                        LOGGER.warning("Failed to optimize structure for block {}.", resname)
                        LOGGER.warning("Proceeding with unoptimized coordinates.")
                        LOGGER.warning("Usually this is OK, but check your final structure.""")
                        break
                    else:
                        opt_counter += 1

                if resname not in self.volumes:
                    self.volumes[resname] = compute_volume(block,
                                                           coords,
                                                           self.topology.nonbond_params)
                coords = map_from_CoG(coords)
                self.templates[resname] = coords

    def run_molecule(self, meta_molecule):
        """
        Execute the generation of templates and set the template
        and volume attribute.
        """
        if hasattr(meta_molecule, "templates"):
            self.templates.update(meta_molecule.templates)
        self.gen_templates(meta_molecule)
        meta_molecule.templates = self.templates
        return meta_molecule
