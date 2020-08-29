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
High level API for the polyply tool to analyze tacticity
"""

import networkx as nx
import vermouth.forcefield
from .apply_links import neighborhood

def analyze_tacticity(chiral_graph):
    adj_matrix = nx.get_adjecy_matrix(chiral_graph)
    chirality_values = nx.get_node_attributes(chiral_graph, "chirality")
    chirality_matrix = chirality_values * chirality_values.reshape((-1, 1))
    adj_chirality = adj_matrix * chirality_matrix
    remain = np.sum(adj_chiraliy[adj_chiraliy == 1.])
    change = np.sum(adj_chiraliy[adj_chiraliy == -1.])
    return remain/(remain+change)

def analyze_tacticity(args):
    """
    Analyze the tacticty of an input structure.
    """
    topology = Topology.from_gmx_topfile(name=args.name, path=args.toppath)
    topology.preprocess()

    # check if molecules are all connected
    for molecule in topology.molecules:
        if not nx.is_connected(molecule):
            msg = ('\n Molecule {} consistes of two disconnected parts. '
                   'Make sure all atoms/particles in a molecule are '
                   'connected by bonds, constraints or virual-sites')
            raise IOError(msg.format(molecule.name))

    # read in coordinates if there are any
    if args.coordpath:
        topology.add_positions_from_file(args.coordpath)
    else:
        for molecule in topology.molecules:
            for node in molecule.molecule.nodes:
                molecule.molecule.nodes[node]["build"] = True

    results = {}
    for molecule in molecules:
       results[molecule.moltype] = _determine_tacticity(molecule, templates)

    pprint()
