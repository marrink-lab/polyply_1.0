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


def create_generic_cores() -> np.ndarray:
    """ """
    pass
