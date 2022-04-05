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

"""
High level API for the polyply DNA coordinate generator
"""
import sys, glob, os
import numpy as np
import vermouth
from vermouth.file_writer import DeferredFileWriter
from vermouth.gmx import read_gro
from vermouth.log_helpers import StyleAdapter, get_logger
from polyply import DATA_PATH
from .generate_templates import GenerateTemplates
from .backmap_dna import Backmap_DNA
from .topology import Topology
from .annotate_dna import AnnotateDNA
from .build_file_parser import read_build_file
from . import coord_file_parser
from .build_system import BuildSystem
from .meta_molecule import MetaMolecule
from .load_library import load_library

LOGGER = StyleAdapter(get_logger(__name__))
BASE_LIBRARY = {"DA": "DT", "DT": "DA", "DG": "DC", "DC": "DG",
                "DA5": "DT3", "DT5": "DA3", "DG5": "DC3", "DC5": "DG3",
                "DA3": "DT5", "DT3": "DA5", "DG3": "DC5", "DC3": "DG5"
                }


def circle_coords(molecule):
    """
    Generate coordinates for molecule so
    that its atoms are placed on a closed circle.

    Parameters
    ----------
    molecule:  :class:vermouth.molecule

    Returns
    -------
    coords : np.ndarray
        array of atom coordinates with shape (len(molecule.nodes), 3)
    """
    nnodes = len(molecule.nodes)
    spacing = 0.33
    radius = spacing / (2 * np.sin(np.pi / nnodes))
    coords = np.zeros([nnodes, 3])
    for ndx in range(nnodes):
        coords[ndx] = [radius * np.sin(2 * np.pi / nnodes * ndx),
                       radius * np.cos(2 * np.pi / nnodes * ndx),
                       0.0]
    return np.array(coords)

def line_coords(molecule):
    """
    Generate coordinates for molecule so that
    its atoms are placed in line on tha x-axis.

    Parameters
    ----------
    molecule:  :class:vermouth.molecule

    Returns
    -------
    coords : np.ndarray
        array of atom coordinates with shape (len(molecule.nodes), 3)
    """
    nnodes = len(molecule.nodes)
    spacing = 0.33
    coords = [[spacing * ndx, 0.0, 0.0] for ndx in range(nnodes)]
    return np.array(coords)

def _read_templates_from_lib(topology, force_field):
        path = os.path.join(DATA_PATH, f"{force_field}/nucleobase_templates/*.gro")
        templates = {}
        for file_ in glob.glob(path):
            base = os.path.basename(file_)[:-4]
            base_template = read_gro(file_)

            templates[base] = {}
            for node in base_template.nodes:
                atomname = base_template.nodes[node]["atomname"]
                position = base_template.nodes[node]["position"]
                templates[base][atomname] = position

        for molecule in topology.molecules:
            molecule.templates = templates

def gen_dna(args):
    LOGGER.info("reading topology",  type="step")
    extension = args.toppath.suffix.casefold()[1:]
    if extension == "top":
        topology = Topology.from_gmx_topfile(name=args.name, path=args.toppath)
    elif extension == 'json':
        force_field = load_library(args.name, ['martini2'], None)
        topology = Topology(force_field=force_field)
        meta_molecule = MetaMolecule.from_json(json_file=args.seqf,
                                               force_field=force_field,
                                               mol_name=args.name)
        topology.molecules = [meta_molecule]
    else:
        raise IOError("Cannot read coordinates from file"\
                      f"with extension .{extension}")

    LOGGER.info("reading build file",  type="step")
    if args.build:
        with open(args.build) as build_file:
            lines = build_file.readlines()
            read_build_file(lines, topology.molecules, topology)

    LOGGER.info("generate/load templates",  type="step")
    if args.generate_templates:
        GenerateTemplates(topology=topology, max_opt=10).run_system(topology)
    else:
        _read_templates_from_lib(topology, args.force_field)

    LOGGER.info("annotating DNA strands",  type="step")
    AnnotateDNA(BASE_LIBRARY).run_system(topology)

    LOGGER.info("generating system coordinates",  type="step")
    for molecule in topology.molecules:
        if args.random_walk:
            # Perform random walk to build system
            pass
        else:
            if args.line:
                positions, normals, tangents = line_coords(molecule), [], []
            elif args.circle:
                positions, normals, tangents = circle_coords(molecule), [], []
            else:
                extension = args.coordpath.suffix.casefold()[1:]
                try:
                    parser = getattr(coord_file_parser, f"read_{extension}")
                    positions, normals, tangents = parser(args.coordpath)
                except AttributeError:
                    raise IOError("Cannot read coordinates from file"\
                                  f"with extension .{extension}")

            for ndx, node in enumerate(molecule.nodes):
                molecule.nodes[node]["position"] = positions[ndx]
                if normals != []:
                    molecule.nodes[node]["normal"] = normals[ndx]
                if tangents != []:
                    molecule.nodes[node]["tangent"] = tangents[ndx]

    LOGGER.info("backmapping to target resolution",  type="step")
    backmap = Backmap_DNA(force_field=args.force_field, fudge_coords=args.bfudge)
    backmap.run_system(topology)

    LOGGER.info("writing output",  type="step")
    command = ' '.join(sys.argv)
    system = topology.convert_to_vermouth_system()
    vermouth.gmx.gro.write_gro(system, args.outpath, precision=7,
                               title=command, box=args.box)
    DeferredFileWriter().write()
