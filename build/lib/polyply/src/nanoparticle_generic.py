import itertools
import logging  # implement logging parts here
from typing import Any, Dict, List, Literal

import networkx as nx
import numpy as np
import vermouth
import vermouth.forcefield
import vermouth.molecule
from polyply.src.generate_templates import (
    _expand_inital_coords,
    _relabel_interaction_atoms,
    extract_block,
    find_atoms,
    replace_defined_interaction,
)
from polyply.src.linalg_functions import center_of_geometry

# import polyply.src.meta_molecule
# from polyply.src.load_library import load_library
from polyply.src.meta_molecule import MetaMolecule
from polyply.src.processor import Processor
from polyply.src.topology import Topology
from pydantic import BaseModel, validator
from scipy.spatial import distance
from vermouth.gmx import gro  # gro library to output the file as a gro file
from vermouth.gmx.itp import write_molecule_itp

# Nanoparticle types
from amber_nps import return_amber_nps_type  # this will be changed
from cg_nps import return_cg_nps_type  # this needs to be changed as well

# logging configuration
logging.basicConfig(level=logging.INFO)


class CentralCoreGenerator:
    """
    Generation of the structure of the center core of the NP.
    Initially, we will be using the f
    """

    def __init__(self, filename, points, R, outputPDB, center):
        self.points = points
        self.R = R
        self.outputPDB = outputPDB
        self.center = center
        self.output = open(filename + ".itp", "w")

    def Nanoparticle_Base_Fibonacci_Sphere(self, samples: int = 1) -> np.ndarray:
        """
        Function to create even points on a sphere for the base of a Nanoparticle.
        """
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians
        for i in range(samples):
            y = (
                1 - (i / float(samples - 1)) * 2
            )  # y goes from 1 to -1 radius = math.sqrt(1 - y * y) # radius at y
            theta = phi * i  # golden angle increment
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            points.append((x, y, z))
        # Return the surface points on the sphere
        self.points = points

    def _create_polyply_object(self) -> Any:
        """ """
        pass


class GoldNanoparticleSingle:
    """
    this class is also embedded with the pydantic model which gives an error in case the
    error

    this is probably rather new so I'm happy to change this back to the original base
    class like structure as with in the other modules

    Parameters
    ----------
    Returns
    -------
    """

    def __init__(self):
        # orientation: Any
        self.lattice_constant: float = 4.0782
        self.box_size: int = 100
        self.matrix: Any = np.array(
            [[self.box_size, 0, 0], [0, self.box_size, 0], [0, 0, self.box_size]]
        )
        self.n_atom_thiol: int = 9
        self.n_phi: int = 8
        self.n_theta: int = 17
        self.n_thiol: int = self.n_phi * self.n_theta
        self.gold_layer: int = 6
        self.n_gold: float = (
            (10 * (self.gold_layer**3) + 11 * self.gold_layer) / 3
            - 5 * (self.gold_layer**2)
            - 1
        )
        # matrix store gold atom coordinates
        self.pos_gold: Any = np.zeros((int(self.n_gold), 3))  # TODO
        self.center: list[float] = [0.5, 0.5, 0.5]  # TODO
        self.phi: float = (np.sqrt(5) + 1) / 2
        self.V: Any  # this should be changed from any to something more appropriate
        self.E: Any  # ditto
        self.F: Any  # ditto
        # blocks and nanoparticle specific parts to work with polyply
        self.force_field: Any  # this needs a custom type
        self.top: Any  # this needs a custom typey
        self.NP_block: Any

    def _set_ff(self) -> None:
        """ """
        logging.info("defining the nanoparticle and ligand block")
        self.NP_block = self.force_field[self.np_name]
        self.ligand_block = self.force_field.blocks[self.ligand_name]

    def _vertices_edges(self) -> None:
        """
        iniiate np array values
        """
        logging.info("defining V E and F")
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

    def _generate_positions(self) -> None:
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

    def create_gro(
        self, outfile: str, velocities: bool = False, box: str = "5.0, 5.0, 5.0"
    ) -> None:
        """
        Plot out the lattice in a 3D structure - generating the coordinates part
        is easy. Now
        the hard part is generating the gro file
        this is somewhat working ... nowhere near what I want - need to review
        the core generation code I made above

        Parameters
        ----------
        outfile: str
        velocities: bool
        box: str

        Returns
        -------
        None

        """
        logging.info("Creating the main gromacs file")
        velocity_fmt = ""
        self._generate_positions()  # generate positions for the gold lattice core
        # logging.info("put log here")
        core_numpy_coords = self.pos_gold
        coords = [[j for j in i] for i in core_numpy_coords]
        gold_str = [f"1AU  AU  {i + 1}" for i in range(0, len(coords))]
        # just need to figure out GRO CONTENT and COORINDATES
        velocities = [[x + 1 for x in line] for line in coords]
        with open(str(outfile), "w") as output:
            output.write("NP\n")
            output.write(str(len(coords)) + "\n")
            for atom, coords, vels in zip(gold_str, coords, velocities):
                output.write(
                    ("{}{:8.3f}{:8.3f}{:8.3f}" + velocity_fmt + "\n").format(
                        atom, *itertools.chain(coords, vels)
                    )
                )

            output.write(box)
            output.write("\n")

    # unfinished part of the code..
    # sampleNPCore = GoldNanoparticleSingle()
    # sampleNPCore._generate_positions()
    # sampleNPCore.create_gro(
    #    "/home/sang/Desktop/example_gold.gro"
    # )  # this works but not fully

    # generate the PCBM nanoparticle in martini
    # -------------------------------------
