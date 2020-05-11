"""
Provides a class used to describe a gromacs topology and all assciated data.
"""
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from vermouth.system import System
from vermouth.forcefield import ForceField
from vermouth.gmx.gro import read_gro
from vermouth.pdb import read_pdb
from .top_parser import read_topology
from .meta_molecule import MetaMolecule
from .linalg_functions import center_of_geometry

COORD_PARSERS = {"pdb": read_pdb,
                 "gro": read_gro}

def find_atoms(molecule, attr, value):
    nodes=[]
    for node in molecule.nodes:
        if attr in molecule.nodes[node]:
           if molecule.nodes[node][attr] == value:
              nodes.append(node)

    return nodes

class Topology(System):
    """
    Ties together vermouth molecule definitions, and
    Gromacs topology information.

    Parameters
    ----------
    force_field: :class:`vermouth.forcefield.ForceField`
        A force field object.
    name: str, optional
        The name of the topology.

    Attributes
    ----------
    molecules: list[:class:`~vermouth.molecule.Molecule`]
        The molecules in the system.
    force_field: a :class:`vermouth.forcefield.ForceField`
    nonbond_params: dict
        A dictionary of all nonbonded parameters
    types: dict
        A dictionary of all typed parameter
    defines: list
        A list of everything that is defined
    """

    def __init__(self, force_field, name=None):
        super().__init__(force_field)
        self.name = name
        self.defaults = {}
        self.defines = {}
        self.description = []
        self.atom_types = {}
        self.types = defaultdict(list)
        self.nonbond_params = {}

    def add_positions_from_gro(self, path):
        """
        Add positions to topology from coordinate file.
        """
        path = Path(path)
        extension = path.suffix.casefold()[1:]
        reader = COORD_PARSERS[extension]
        molecules = read_gro(path, exclude=())
        total = 0
        for meta_mol in self.molecules:
            for node in meta_mol.molecule.nodes:
                try:
                   position = molecules.nodes[total]["position"]
                except KeyError:
                   meta_mol.molecule.nodes[node]["build"] = True
                   last_atom = total
                else:
                   meta_mol.molecule.nodes[node]["position"] = position
                   meta_mol.molecule.nodes[node]["build"] = False
                   total += 1

            for node in meta_mol:
                atoms_in_res = find_atoms(meta_mol.molecule, "resid", node+1)
                if last_atom not in atoms_in_res:
                   positions = np.array([ meta_mol.molecule.nodes[atom]["position"] for
                                         atom in atoms_in_res])
                   center = center_of_geometry(positions)
                   meta_mol.nodes[node]["position"] = center
                else:
                   break

    def convert_to_vermouth_system(self):
        system = System()
        system.molecules = []
        system.force_field = self.force_field

        for meta_mol in self.molecules:
            system.molecules.append(meta_mol.molecule)

        return system

    @classmethod
    def from_gmx_topfile(cls, path, name):
        """
        Read a gromacs topology file and return an topology object.

        Parameters
        ----------
        path:  str
           The name of the topology file
        name:  str
           The name of the system
        """
        with open(path, 'r') as _file:
            lines = _file.readlines()

        cwdir = os.path.dirname(path)
        force_field = ForceField(name)
        topology = cls(force_field=force_field, name=name)
        read_topology(lines=lines, topology=topology, cwdir=cwdir)
        return topology
