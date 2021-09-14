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
    Probability to find an end-to-end distance h given a contour length
    L and presistence length _lambda according to the Worm-Like-Chain
    model. The equation was published in: "Lee, H. and Pastor, R.W., 2011.
    Coarse-grained model for PEGylated lipids: effect of PEGylation on the
    size and shape of self-assembled structures. The Journal of Physical
    Chemistry B, 115(24), pp.7830-7837."

    Parameters
    ----------
    h: float
        the end-to-end distance
    L: float
        the contour length
    _lambda: float
        the presistence length

    Returns
    -------
    float
        probability to find the end-to-end distance
    """
    alpha = 3*L/(4*_lambda)
    C = (np.pi**3/2. * np.exp(-alpha)*alpha**(-3/2.)*(1 + 3/(alpha) + (15/4)/alpha**2.))**-1. 
    A = 4*np.pi*h**2.*C
    B = L*(1-(h/L)**2.)**9/2.
    D = -3*L
    E = 4*_lambda*(1-(h/L)**2.)
    return A/B * np.exp(D/E)


DISTRIBUTIONS = {"WCM": worm_like_chain_model, }

def generate_end_end_distances(molecule, specs, nonbond_matrix, seed=None):
    """
    Subsample a distribution of end-to-end distances given a
    presistence length, residue graph, and theoretical model.

    Parameters
    ----------
    molecule: `class:nx.Graph`
    specs: `tuple`
        named tuple with attributes model, lp (i.e. presistence length),
        start, stop node indices, which define the ends and mol_idxs
        which are the indices of the batch of molecules.
    nonbond_matrix: `:class:polyply.src.nb_engine.NonBondMatrix`
    seed: int
        random seed for subsampling distribution

    Returns
    -------
    np.ndarray
        an array of end-to-end distances with shape (1, len(mol_idxs))
    """
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
    # define range of end-to-end distances
    # increment is the average step length
    incrm = max_path_length / len(end_to_end_path)
    ee_distances = np.arange(incrm, max_path_length, incrm)
    np.random.seed(seed)
    # generate the distribution and sample it
    probabilities = DISTRIBUTIONS[specs.model](ee_distances, max_path_length, specs.lp)
    #TODO
    #replace choice by https://numpy.org/doc/stable/reference/random/generated/
    #                          numpy.random.Generator.choice.html#numpy.random.Generator.choice
    np.random.seed(seed)
    ee_samples = np.random.choice(ee_distances,
                                  p=probabilities/np.sum(probabilities),
                                  size=len(specs.mol_idxs))
    return ee_samples

def sample_end_to_end_distances(molecules, topology, nonbond_matrix, seed=None):
    """
    Apply distance restraints to the ends of molecules given a distribution
    of end-to-end distances generated from a given theoreical model. The
    topology attribute presistences contains a list of batches of molecules
    for which a presistence length has been given. This function generates
    the corresponding restraints taking into account the actual path
    lengths as defined by the volumes in the super-CG model.

    Parameters
    ----------
    molecules: list[`:class:nx.Graph`]
        the molecules as graphs
    topology: `class:polyply.src.topology.Topology`
        the topology of the system
    nonbond_matrix: `:class:polyply.src.nb_engine.NonBondMatrix`
        the nonbonded matrix that stores the interactions of the
        super CG model built from the topology
    seed:
        random seed for subsampling distribution

    Returns
    -------
    list[`:class:nx.Graph`]
        list of updated molecules
    """
    # loop over all batches of molecules
    for specs in topology.presistences:
        molecule = molecules[specs.mol_idxs[0]]
        # generate a distribution of end-to-end distances
        distribution = generate_end_end_distances(molecule,
                                                  specs,
                                                  nonbond_matrix,
                                                  seed=seed)
        # for each molecule in the batch assign one end-to-end
        # distance restraint from the distribution.
        for mol_idx, dist in zip(specs.mol_idxs, distribution):
            apply_node_distance_restraints(molecules[mol_idx],
                                           specs.stop,
                                           dist,
                                           ref_node=specs.start)
    return molecules
