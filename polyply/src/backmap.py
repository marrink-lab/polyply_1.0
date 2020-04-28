import numpy as np
import networkx as nx
from .processor import Processor
from .generate_templates import find_atoms


class Backmap(Processor):


    def _place_init_coords(self, meta_molecule):
        coords = np.zeros((len(meta_molecule.molecule.nodes), 3))
        idx = 0
        for node in meta_molecule.nodes:
            resname = meta_molecule.nodes[node]["resname"]
            CoG = meta_molecule.nodes[node]["position"]
            template = meta_molecule.templates[resname]
            resid = node + 1
            low_res_atoms = find_atoms(meta_molecule.molecule, "resid", resid)

            for atom_super, atom_low  in zip(template,low_res_atoms):
                vector = template[atom_super]
                new_coords = CoG + vector
                if meta_molecule.molecule.nodes[atom_low]["build"]:
                   meta_molecule.molecule.nodes[atom_low]["position"] = new_coords

    def run_molecule(self, meta_molecule):
        self._place_init_coords(meta_molecule)
        return meta_molecule
