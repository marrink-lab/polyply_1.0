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
High level API for the polyply coordinate generator
"""

from pathlib import Path
import networkx as nx
import vermouth.forcefield
from vermouth.gmx.gro import read_gro
from polyply import (MetaMolecule, DATA_PATH)
from .generate_templates import GenerateTemplates
from .random_walk import RandomWalk
from .backmap import Backmap
from .minimizer import optimize_geometry
from .topology import Topology

def gen_coords(args):
    # Read in the topology
    topology = Topology.from_gmx_topfile(name=args.name, path=args.toppath)
    topology.gen_pairs()
    topology.replace_defins()
    # convert all parameters to sigma epsilon if they are in C6C12 form
    if topology.defaults["comb-rule"] == 2:
       topology.convert_nonbond_to_sig_eps()

    if args.coordpath:
       topology.add_positions_from_file(args.coordpath)
    else:
       for molecule in topology.molecules:
           for node in molecule.molecule.nodes:
               molecule.molecule.nodes[node]["build"] = True

    # Build polymer structure
    GenerateTemplates().run_system(topology)
    RandomWalk().run_system(topology)
    Backmap().run_system(topology)
    #energy_minimize().run_system(topology)

    system = topology.convert_to_vermouth_system()
    # Write output
    vermouth.gmx.gro.write_gro(system, args.outpath, precision=7,
              title='polyply structure', box=(10, 10, 10))
