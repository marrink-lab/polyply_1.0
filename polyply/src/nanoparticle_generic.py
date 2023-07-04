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
