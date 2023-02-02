# testing out dataclasse
from dataclasses import dataclass

import networkx as nx
import numpy as np
import vermouth
from scipy.spatial import distance

import polyply.src.meta_molecule
from polyply.src.generate_templates import GenerateTemplates
from polyply.src.processor import Processor
from polyply.src.topology import Topology
from polyply.src.load_library import load_library

import vermouth.forcefield
import vermouth.molecule

# import textwrap
# from pathlib import Path


"""
Internal notes:
--------------

for the UAModelSingleNP:

The program outputs LAMMPS data file "data.np" and CFG configuration file 
cfg which can be visualized with the atomeeye file 

The program reads the relaxed coordinates of a single gold-alkanethiol
nanoparticle. These coordinates are writte in the file 
singleNP.cfg. 

-- notes from singleNP: 

The program outputs LAMMPS data file "data.np" and CFG file. 

"""
# generate pydantic class for lennard jones parameters
# pair_coeff       1 1 morse  10.956 1.5830 3.0242 # These paramaeters are identical as with the other orientations and visa versa


class NanoparticleSingle:
    """

    This program is
    adopting the single nanoparticle code into a python dataclass.
    """

    def __init__(self):
        self.orientation = None
        self.lattice_constant: float = 4.0782
        self.box_size: int = 100
        self.matrix: np.ndarray = np.array(
            [[self.box_size, 0, 0], [0, self.box_size, 0], [0, 0, self.box_size]]
        )

        # number of atoms per thiol in united atom model
        self.n_atom_thiol: int = 9

        # the aziimuthal angle and polar angle are divided evenely
        # forming a grid, the thiol molecules are then put on these grids.
        # The total number of thiol molecules will be the product of nphi
        # and ntheta
        self.n_phi: int = 8
        self.n_theta: int = 17
        self.n_thiol: int = self.n_phi * self.n_theta

        # layer of gol atoms in gold icosahedral nanocrystal
        # different number of gold layers wil lreult in
        # different number of gold atoms in the nanocrystal

        # gold layer = 6
        self.gold_layer: int = 6
        self.n_gold: float = (
            (10 * (self.gold_layer**3) + 11 * self.gold_layer) / 3
            - 5 * (self.gold_layer**2)
            - 1
        )

        # matrix store gold atom coordinates
        self.pos_gold: np.ndarray = np.zeros((int(self.n_gold), 3))  # TODO
        self.center: list[float] = [0.5, 0.5, 0.5]  # TODO
        self.phi: float = (np.sqrt(5) + 1) / 2

    def _vertices_edges(self):
        """
        iniiate np array values
        """
        self.V = np.zeros((12, 3))
        self.E = np.zeros((2, 3, 30))
        self.F = np.zeros((3, 3, 20))

        self.V[0, :] = [1, 0, self.phi]
        self.V[1, :] = [-1, 0, self.phi]
        self.V[2, :] = [-1, self.phi, 1]
        self.V[3, :] = [self.phi, -1, 0]
        self.V[4, :] = [self.phi, 1, 0]
        self.V[5, :] = [0, self.phi, 1]
        # -ve persions of the phi values
        self.V[6:, :] = -self.V[:6, :]

        self.E[0, :, 0] = [1, 0, self.phi]
        self.E[0, :, 1] = [1, 0, self.phi]
        self.E[0, :, 2] = [1, 0, self.phi]
        self.E[0, :, 3] = [1, 0, self.phi]
        self.E[0, :, 4] = [1, 0, self.phi]
        self.E[0, :, 5] = [0, self.phi, 1]
        self.E[0, :, 6] = [0, self.phi, 1]
        self.E[0, :, 7] = [0, self.phi, 1]
        self.E[0, :, 8] = [0, self.phi, 1]
        self.E[0, :, 9] = [self.phi, 1, 0]
        self.E[0, :, 10] = [self.phi, 1, 0]
        self.E[0, :, 11] = [self.phi, 1, 0]
        self.E[0, :, 13] = [1, 0, -self.phi]
        self.E[0, :, 14] = [0, self.phi, -1]
        self.E[1, :, 0] = [-1, 0, self.phi]
        self.E[1, :, 1] = [0, -self.phi, 1]
        self.E[1, :, 2] = [self.phi, -1, 0]
        self.E[1, :, 3] = [self.phi, 1, 0]
        self.E[1, :, 4] = [0, self.phi, 1]
        self.E[1, :, 5] = [self.phi, 1, 0]
        self.E[1, :, 6] = [0, self.phi, -1]
        self.E[1, :, 7] = [-self.phi, 1, 0]
        self.E[1, :, 8] = [-1, 0, self.phi]
        self.E[1, :, 9] = [self.phi, -1, 0]
        self.E[1, :, 10] = [1, 0, -self.phi]
        self.E[1, :, 11] = [0, self.phi, -1]
        self.E[1, :, 12] = [1, 0, -self.phi]
        self.E[1, :, 13] = [0, self.phi, -1]
        self.E[1, :, 14] = [-self.phi, 1, 0]
        self.E[:, :, 15:] = -self.E[:, :, :15]

        self.F[1, :, 0] = [1, 0, self.phi]
        self.F[1, :, 0] = [-1, 0, self.phi]
        self.F[2, :, 0] = [0, -self.phi, 1]
        self.F[0, :, 1] = [1, 0, self.phi]
        self.F[1, :, 1] = [0, -self.phi, 1]
        self.F[2, :, 1] = [self.phi, -1, 0]
        self.F[0, :, 2] = [1, 0, self.phi]
        self.F[1, :, 2] = [self.phi, -1, 0]
        self.F[2, :, 2] = [self.phi, 1, 0]
        self.F[0, :, 3] = [1, 0, self.phi]
        self.F[1, :, 3] = [self.phi, 1, 0]
        self.F[2, :, 3] = [0, self.phi, 1]
        self.F[0, :, 4] = [1, 0, self.phi]
        self.F[1, :, 4] = [0, self.phi, 1]
        self.F[2, :, 4] = [-1, 0, self.phi]
        self.F[0, :, 5] = [0, self.phi, 1]
        self.F[1, :, 5] = [self.phi, 1, 0]
        self.F[2, :, 5] = [0, self.phi, -1]
        self.F[0, :, 6] = [0, self.phi, 1]
        self.F[1, :, 6] = [0, self.phi, -1]
        self.F[2, :, 6] = [-self.phi, 1, 0]
        self.F[0, :, 7] = [0, self.phi, 1]
        self.F[1, :, 7] = [-self.phi, 1, 0]
        self.F[2, :, 7] = [-1, 0, self.phi]
        self.F[0, :, 8] = [self.phi, 1, 0]
        self.F[1, :, 8] = [self.phi, -1, 0]
        self.F[2, :, 8] = [1, 0, -self.phi]
        self.F[0, :, 9] = [self.phi, 1, 0]
        self.F[1, :, 9] = [1, 0, -self.phi]
        self.F[2, :, 9] = [0, self.phi, -1]
        self.F[:, :, 10:] = -self.F[:, :, :10]

        scaling = self.lattice_constant / np.sqrt(2 * (1 + self.phi * self.phi))
        self.V = self.V * scaling
        self.E = self.E * scaling
        self.F = self.F * scaling

    def generate_positions(self):
        """
        Need some general notes on how the core was generated.
        """
        self._vertices_edges()  # need brief description of this function does
        self.pos_gold[0, :] = self.center

        # coordinates of the second innermost layer atoms
        for i in range(12):
            self.pos_gold[i + 1, 0] = self.V[i, 0] / self.box_size + self.center[0]
            self.pos_gold[i + 1, 1] = self.V[i, 1] / self.box_size + self.center[1]
            self.pos_gold[i + 1, 2] = self.V[i, 2] / self.box_size + self.center[2]

        # coordinates of the third innermost layer atoms
        for i in range(12):
            self.pos_gold[i + 13, 0] = self.V[i, 0] * 2 / self.box_size + self.center[0]
            self.pos_gold[i + 13, 1] = self.V[i, 1] * 2 / self.box_size + self.center[1]
            self.pos_gold[i + 13, 2] = self.V[i, 2] * 2 / self.box_size + self.center[2]

        # coordinates of the fourth innermost layer atoms
        for i in range(30):
            self.pos_gold[i + 25, 0] = (
                self.E[0, 0, i] + self.E[1, 0, i]
            ) / self.box_size + self.center[0]
            self.pos_gold[i + 25, 1] = (
                self.E[0, 1, i] + self.E[1, 1, i]
            ) / self.box_size + self.center[1]
            self.pos_gold[i + 25, 2] = (
                self.E[0, 2, i] + self.E[1, 2, i]
            ) / self.box_size + self.center[2]

        for N in range(3, self.gold_layer):

            n_atom_to_now = int(
                ((10 * (N - 1) ** 3 + 11 * (N - 1)) / 3 - 5 * ((N - 1) ** 2) - 1)
            )

            for i in range(12):
                self.pos_gold[n_atom_to_now + i][0] = (
                    self.V[i][0] * (N - 1) / self.box_size + self.center[0]
                )
                self.pos_gold[n_atom_to_now + i][1] = (
                    self.V[i][1] * (N - 1) / self.box_size + self.center[1]
                )
                self.pos_gold[n_atom_to_now + i][2] = (
                    self.V[i][2] * (N - 1) / self.box_size + self.center[2]
                )
            n_atom_to_now = n_atom_to_now + 12

            for i in range(30):
                for j in range(N - 2):
                    self.pos_gold[n_atom_to_now + (i - 1) * (N - 2) + j][0] = (
                        self.E[0][0][i] * (N - 1)
                        + (self.E[1][0][i] - self.E[0][0][i]) * j
                    ) / self.box_size + self.center[0]
                    self.pos_gold[n_atom_to_now + (i - 1) * (N - 2) + j][1] = (
                        self.E[0][1][i] * (N - 1)
                        + (self.E[1][1][i] - self.E[0][1][i]) * j
                    ) / self.box_size + self.center[1]
                    self.pos_gold[n_atom_to_now + (i - 1) * (N - 2) + j][2] = (
                        self.E[0][2][i] * (N - 1)
                        + (self.E[1][2][i] - self.E[0][2][i]) * j
                    ) / self.box_size + self.center[2]

            n_atom_to_now = n_atom_to_now + 30 * (N - 2)

            for i in range(20):
                for m in range(N - 3):
                    for n in range(N - m - 2):
                        self.pos_gold[
                            n_atom_to_now + (2 * N - m - 4) * (m - 1) // 2 + n, 0
                        ] = (
                            self.F[0, 0, i] * (N - 1)
                            + (self.F[1, 0, i] * (N - 1) - self.F[0, 0, i] * (N - 1))
                            * m
                            / (N - 1)
                            + (self.F[2, 0, i] * (N - 1) - self.F[0, 0, i] * (N - 1))
                            * n
                            / (N - 1)
                        ) / self.box_size + self.center[
                            0
                        ]
                        self.pos_gold[
                            n_atom_to_now + (2 * N - m - 4) * (m - 1) // 2 + n, 1
                        ] = (
                            self.F[0, 1, i] * (N - 1)
                            + (self.F[1, 1, i] * (N - 1) - self.F[0, 1, i] * (N - 1))
                            * m
                            / (N - 1)
                            + (self.F[2, 1, i] * (N - 1) - self.F[0, 1, i] * (N - 1))
                            * n
                            / (N - 1)
                        ) / self.box_size + self.center[
                            1
                        ]
                        self.pos_gold[
                            n_atom_to_now + (2 * N - m - 4) * (m - 1) // 2 + n, 2
                        ] = (
                            self.F[0, 2, i] * (N - 1)
                            + (self.F[1, 2, i] * (N - 1) - self.F[0, 2, i] * (N - 1))
                            * m
                            / (N - 1)
                            + (self.F[2, 2, i] * (N - 1) - self.F[0, 2, i] * (N - 1))
                            * n
                            / (N - 1)
                        ) / self.box_size + self.center[
                            2
                        ]
                n_atom_to_now += (N - 2) * (N - 3) // 2


class NanoparticleLatticeGenerator(Processor):
    """
    analyzing and creating the class depending on the
    number of atoms of gold we want

    """

    def __init__(self, force_field):
        self.force_field = force_field

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
        return


if __name__ == "__main__":
    sampleNPCore = NanoparticleSingle()
    sampleNPCore.generate_positions()
    # test amber force field loading libraries
    # force_field = load_library("gold", ["amber"], inpath)
    gold = vermouth.gmx.read_gro(
        "/home/sang/Desktop/git/polyply_1.0/polyply/data/nanoparticle_test/AMBER_AU/au102.gro"
    )
