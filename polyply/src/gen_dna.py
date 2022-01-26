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
from vermouth.file_writer import DeferredFileWriter
from vermouth.gmx import read_gro
from vermouth.log_helpers import StyleAdapter, get_logger
from polyply import DATA_PATH
from .generate_templates import GenerateTemplates
from .backmap_dna import Backmap_DNA
from .topology import Topology
from .annotate_dna import AnnotateDNA
from .build_file_parser import read_build_file

LOGGER = StyleAdapter(get_logger(__name__))

def read_xyz(path):
    """
    Read an xyz file and extract molecule's coordinates.

    The file should be in the format::

        <number of atoms>
        comment line
        <element0> <X0> <Y0> <Z0>
        <element1> <X1> <Y1> <Z1>
        ...

    Parameters
    ----------
    path : str
        Path to the xyz file to read

    Returns
    -------
    coords : list
        list of atom coordinates
    """

    coords = []
    with open(path, 'r') as f:
        num_atoms = int(f.readline())
        next(f)

        for line in f:
            _, x, y, z = line.strip().split()
            coords.append([float(x), float(y), float(z)])

        assert num_atoms == len(coords), \
            '<number of atoms> not equal to number of coordinates in xyz file'
    return coords

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
    return coords

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
    coords = [[0.33*ndx, 0, 0] for ndx in range(nnodes)]
    return coords

def gen_dna(args):
    LOGGER.info("reading topology",  type="step")
    topology = Topology.from_gmx_topfile(name=args.name, path=args.toppath)

    LOGGER.info("reading build file",  type="step")
    if args.build:
        LOGGER.info("reading build file",  type="step")
        with open(args.build) as build_file:
            lines = build_file.readlines()
            read_build_file(lines, topology.molecules, topology)

    LOGGER.info("generating/load templates",  type="step")
    if args.generate_templates:
        GenerateTemplates(topology=topology, max_opt=10).run_system(topology)
    else:
        path = DATA_PATH + "/parmbsc1/"
        templates = {}
        for file in glob.glob(path + '*.gro'):
            base = os.path.basename(file)[:-4]
            base_template = read_gro(file)

            templates[base] = {}
            for node in base_template.nodes:
                atomname = base_template.nodes[node]['atomname']
                position = base_template.nodes[node]['position']
                templates[base][atomname] = position

        for molecule in topology.molecules:
            molecule.templates = templates

    LOGGER.info("annotating DNA strands",  type="step")
    dna_annotator = AnnotateDNA(topology, args.includes_DNA)
    dna_annotator.run_system(topology)

    LOGGER.info("generating system coordinates",  type="step")

    for molecule in topology.molecules:
        if args.line:
            coords = line_coords(molecule)
        elif args.circle:
            coords = circle_coords(molecule)
        else:
            coords = read_xyz(args.coordpath)
        for ndx, node in enumerate(molecule.nodes):
            molecule.nodes[node]['position'] = coords[ndx]

    LOGGER.info("backmapping to target resolution",  type="step")
    Backmap_DNA(fudge_coords=args.bfudge).run_system(topology)

    LOGGER.info("writing output",  type="step")
    command = ' '.join(sys.argv)
    system = topology.convert_to_vermouth_system()
    vermouth.gmx.gro.write_gro(system, args.outpath, precision=7,
                               title=command, box=args.box)
    DeferredFileWriter().write()
