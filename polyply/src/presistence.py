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
import numpy as np
import networkx as nx
from .graph_utils import _compute_path_length_cartesian
from .build_file_parser import apply_node_distance_restraints

def worm_like_chain_model(h, L, _lambda):
    """
    worm like chain model distribution of the end to end distance.
    """
    alpha = 3*L/(4*_lambda)
    C = (np.pi**3/2. * np.exp(-alpha)*alpha**(-3/2.)*(1 + 3/(alpha) + (15/4)/alpha**2.))**-1. 
    A = 4*np.pi*h**2.*C
    B = L*(1-(h/L)**2.)**9/2.
    D = -3*L
    E = 4*_lambda*(1-(h/L)**2.)
    
    return A/B * np.exp(D/E)


DISTRIBUTIONS = {"WCM": worm_like_chain_model, }

def generate_end_end_distances(molecule, specs, nonbond_matrix):
    # compute the shortest path between the ends in graph space
    end_to_end_path = nx.algorithms.shortest_path(molecule,
                                                  source=specs.start,
                                                  target=specs.stop)
    end_to_end_path = list(zip(end_to_end_path[:-1], end_to_end_path[1:]))
    # compute the length of that path in cartesian space
    mol_idx = specs.mol_idxs[0]
    max_path_length = _compute_path_length_cartesian(molecule,
                                                     mol_idx,
                                                     end_to_end_path,
                                                     nonbond_matrix)
    # define reange of end-to-end distances
    # increment is the average step length
    incrm = max_path_length / len(end_to_end_path)
    ee_distances = np.arange(incrm, max_path_length, incrm)

    # generate the distribution and sample it
    probabilities = DISTRIBUTIONS[specs.model](ee_distances, max_path_length, specs.lp)
    ee_samples = np.random.choice(ee_distances,
                                  p=probabilities/np.sum(probabilities),
                                  size=len(specs.mol_idxs))
    return ee_samples

def sample_end_to_end_distances(molecules, topology, nonbond_matrix):
    for specs in topology.presistences:
        molecule = molecules[specs.mol_idxs[0]]
        distribution = generate_end_end_distances(molecule,
                                                  specs,
                                                  nonbond_matrix)

        for mol_idx, dist in zip(specs.mol_idxs, distribution):
            apply_node_distance_restraints(molecules[mol_idx],
                                           specs.start,
                                           specs.stop,
                                           dist)
    return molecules
