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
from vermouth.log_helpers import StyleAdapter, get_logger
from .graph_utils import compute_avg_step_length, get_all_predecessors
from .restraints import set_distance_restraint

LOGGER = StyleAdapter(get_logger(__name__))

def worm_like_chain_model(ee_dist, r_max, persistence_length):
    """
    Probability to find an end-to-end distance h given a contour length
    L and persistence length _lambda according to the Worm-Like-Chain
    model. The equation was published in: "Lee, H. and Pastor, R.W., 2011.
    Coarse-grained model for PEGylated lipids: effect of PEGylation on the
    size and shape of self-assembled structures. The Journal of Physical
    Chemistry B, 115(24), pp.7830-7837."

    Parameters
    ----------
    ee_dist: float
        the end-to-end distance
    r_max: float
        the contour length
    _lambda: float
        the persistence length

    Returns
    -------
    float
        probability to find the end-to-end distance
    """
    alpha = 3*r_max/(4* persistence_length)
    factor_1 = (np.pi**3/2. * np.exp(-alpha)*alpha**(-3/2.)*(1 + 3/(alpha) + (15/4)/alpha**2.))**-1.
    nominator = 4*np.pi*ee_dist**2.* factor_1

    denominator = r_max*(1-(ee_dist/r_max)**2.)**9/2.
    nominator_exp = -3*r_max
    denominator_exp = 4* persistence_length*(1-(ee_dist/r_max)**2.)

    return nominator/denominator * np.exp(nominator_exp/denominator_exp)


DISTRIBUTIONS = {"WCM": worm_like_chain_model, }

def generate_end_end_distances(specs,
                               avg_step_length,
                               max_path_length,
                               box,
                               limit_prob=1e-5,
                               seed=None):
    """
    Subsample a distribution of end-to-end distances given a
    persistence length, residue graph, and theoretical model.

    Parameters
    ----------
    specs: `tuple`
        named tuple with attributes model, lp (i.e. persistence length),
        start, stop node indices, which define the ends and mol_idxs
        which are the indices of the batch of molecules.
    avg_step_length: float
        average step length (nm)
    max_path_length: float
        total length of the path (nm)
    box: np.ndarray
        box vectors
    limit_prob: float
        the lowest probability after which to truncate end-to-end distance
        distribution
    seed: int
        random seed for subsampling distribution

    Returns
    -------
    np.ndarray
        an array of end-to-end distances with shape (1, len(mol_idxs))
    """
    ee_distances = np.arange(avg_step_length, max_path_length, avg_step_length)
    # generate the distribution and sample it
    probabilities = DISTRIBUTIONS[specs.model](ee_distances, max_path_length, specs.lp)
    # filter the end-to-end distances
    ee_distances = ee_distances[probabilities > limit_prob]
    probabilities = probabilities[probabilities > limit_prob]
    #TODO
    #replace choice by https://numpy.org/doc/stable/reference/random/generated/
    #                          numpy.random.Generator.choice.html#numpy.random.Generator.choice
    np.random.seed(seed)
    ee_samples = np.random.choice(ee_distances,
                                  p=probabilities/np.sum(probabilities),
                                  size=len(specs.mol_idxs))
    # raise warning if the smallest box vector is smaller than the largest end-to-end
    # distance
    if max(ee_samples) > np.sqrt(3*min(box)**2):
        msg = ("Sampling the end-to-end distance distribution yielded end-to-end distances\n"
               "that are larger than the box diagonal. This will not work. Please increase the\n"
               "box to be at least {} nm.")
        raise IOError(msg.format(max(ee_samples)))

    if max(ee_samples) > min(box):
        msg = ("Your smallest box vector {} is smaller than the largest end-to-end distance {},\n"
               "which is dictated by the set persistence length. This can prevent the algorithm \n"
               "from converging, since it will not be able to place the molecule in the \n"
               "relatively small box dimensions. You should increase the boxsize to be at \n"
               "least {} ideally even a bit larger.")
        # add a 10% buffer just to be sure
        LOGGER.warning(msg, min(box), max(ee_samples), max(ee_samples)*1.1)

    return ee_samples

def sample_end_to_end_distances(topology, nonbond_matrix, seed=None):
    """
    Apply distance restraints to the ends of molecules given a distribution
    of end-to-end distances generated from a given theoreical model. The
    topology attribute persistences contains a list of batches of molecules
    for which a persistence length has been given. This function generates
    the corresponding restraints taking into account the actual path
    lengths as defined by the volumes in the super-CG model.

    Parameters
    ----------
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
    for specs in topology.persistences:
        molecule = topology.molecules[specs.mol_idxs[0]]

        # first we initalize the path
        molecule.root = specs.start
        path = get_all_predecessors(molecule.search_tree,
                                    start_node=specs.start,
                                    node=specs.stop)
        edge_path = list(zip(path[:-1], path[1:]))
        # compute the average step length end-to-end
        avg_step_length, max_length = compute_avg_step_length(molecule,
                                                              specs.mol_idxs[0],
                                                              nonbond_matrix,
                                                              edge_path)
        # generate a distribution of end-to-end distances
        distribution = generate_end_end_distances(specs,
                                                  avg_step_length,
                                                  max_length,
                                                  box=nonbond_matrix.boxsize,
                                                  seed=seed)

        # for each molecule in the batch assign one end-to-end
        # distance restraint from the distribution.
        for mol_idx, dist in zip(specs.mol_idxs, distribution):
            # we set the bfs root here to enforce that start
            # is placed before stop
            topology.molecules[mol_idx].root = specs.start
            set_distance_restraint(topology.molecules[mol_idx],
                                   specs.stop,
                                   specs.start,
                                   dist,
                                   avg_step_length,
                                   tolerance=0.0)
