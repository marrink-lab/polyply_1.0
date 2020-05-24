import numpy as np
import networkx as nx
from .processor import Processor
from .generate_templates import find_atoms


class Backmap(Processor):
    """
    This processor takes a a class:`polyply.src.MetaMolecule` and
    places coordinates form a higher resolution createing positions
    for the lower resolution molecule associated with the MetaMolecule.
    """

    def _place_init_coords(self, meta_molecule):
        """
        For each residue in a class:`polyply.src.MetaMolecule` the
        positions of the atoms associated with that residue stored in
        attr:`polyply.src.MetaMolecule.molecule` are created from a
        template residue located in attr:`polyply.src.MetaMolecule.templates`.

        Parameters
        ----------
        class:`polyply.src.MetaMolecule`
        """
        coords = np.zeros((len(meta_molecule.molecule.nodes), 3))
        idx = 0
        for node in meta_molecule.nodes:
            resname = meta_molecule.nodes[node]["resname"]
            CoG = meta_molecule.nodes[node]["position"]
            template = meta_molecule.templates[resname]
            resid = node + 1
            low_res_atoms = find_atoms(meta_molecule.molecule, "resid", resid)

            for atom_super, atom_low  in zip(template, low_res_atoms):
                vector = template[atom_super]
                new_coords = CoG + vector
                if meta_molecule.molecule.nodes[atom_low]["build"]:
                   meta_molecule.molecule.nodes[atom_low]["position"] = new_coords

    def run_molecule(self, meta_molecule):
        """
        Apply placing of coordinates to meta_molecule. For more
        detail see `self._place_init_coords`.
        """
        self._place_init_coords(meta_molecule)
        return meta_molecule
