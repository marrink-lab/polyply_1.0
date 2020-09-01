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

import networkx as nx
import vermouth.forcefield
from vermouth.file_writer import open, DeferredFileWriter
from .generate_templates import GenerateTemplates
from .random_walk import RandomWalk
from .backmap import Backmap
from .topology import Topology

def gen_coords(args):
    # Read in the topology
    print("loading topology")
    topology = Topology.from_gmx_topfile(name=args.name, path=args.toppath)
    topology.preprocess()

    # check if molecules are all connected
    for molecule in topology.molecules:
        if not nx.is_connected(molecule):
            msg = ('\n Molecule {} consistes of two disconnected parts. '
                   'Make sure all atoms/particles in a molecule are '
                   'connected by bonds, constraints or virual-sites')
            raise IOError(msg.format(molecule.name))

    # renumber the resiudes so that all molecules
    # start with residue index 1
    for meta_molecule in topology.molecules:
        molecule = meta_molecule.molecule
        resids = nx.get_node_attributes(molecule, "resid")
        offset = min(resids.values()) - 1
        new_resids = {node: resid - offset for node, resid in resids.items()}
        nx.set_node_attributes(molecule, new_resids, "resid")

    # read in coordinates if there are any
    if args.coordpath:
        topology.add_positions_from_file(args.coordpath)
    else:
        for molecule in topology.molecules:
            for node in molecule.nodes:
                molecule.nodes[node]["build"] = True

    # Build polymer structure
    GenerateTemplates().run_system(topology)
    RandomWalk().run_system(topology)
    Backmap().run_system(topology)
    #energy_minimize().run_system(topology)

    system = topology.convert_to_vermouth_system()
    # Write output
    vermouth.gmx.gro.write_gro(system, args.outpath, precision=7,
                               title='polyply structure', box=(10, 10, 10))
    DeferredFileWriter().write()
