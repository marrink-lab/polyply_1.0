import numpy as np
from scipy.spatial import distance
import networkx as nx
import vermouth

# import textwrap
# from pathlib import Path

import polyply.src.meta_molecule
from polyply.src.topology import Topology
from polyply.src.processor import Processor
from polyply.src.generate_templates import GenerateTemplates

# testing out dataclasse
from dataclasses import dataclass


"""
Internal notes:
--------------

for the UAModelSingleNP:

The program outputs LAMMPS data file "data.np" and CFG configuration file 
cfg which can be visualized with the atomeeye file 

The program reads the relaxed coordinates of a single gold-alkanethiol
nanoparticle. These coordinates are writte in the file 
singleNP.cfg. 


"""


@dataclass
class NaanoparticleSingle(Processor):
    """

    This program is
    adopting the single nanoparticle code into a python dataclass.
    """

    lattice_constant: float = 4.0782
    box_size: int = 100
    matrix: list[int]
    # number of atoms per thiol in united atom model
    n_atom_thiol: int

    # the aziimuthal angle and polar angle are divided evenely
    # forming a grid, the thiol molecules are then put on these grids.

    # The total number of thiol molecules will be the product of nphi
    # and ntheta
    n_phi: int
    n_theta: int

    # layer of gol atoms in gold icosahedral nanocrystal
    # different number of gold layers wil lreult in
    # different number of gold atoms in the nanocrystal
    # gold layer = 6

    gold_layer: int
    n_gold: float = (
        (10 * (gold_layer**3) + 11 * gold_layer) / 3 - 5 * (gold_layer**2) - 1
    )

    # matrix store gold atom coordinates
    pos_gold: np.array = np.zeros()  # TODO
    center: list[float] = [0.5, 0.5, 0.5]  # TODO


class NanoparticleLatticeGenerator(Processor):
    """
    analyzing and creating the class depending on the
    number of atoms of gold we want

    """

    input_file: str()
    n_atoms_gold: int = 561
    n_atoms_S: int = 136
    n_atom_C1: int = 952
    n_atom_C2: int = 136  # number of CH3 per nanoparticle?

    def generate_coordinates(self) -> None:
        """
        I don't fully know the return type for this yet

        ideally, we generate the coordinates with this function
        and then store it within a 'coordinates' object. The form
        of which I am not certain yet.
        """
        pass
