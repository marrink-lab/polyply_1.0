try:
    import sys
    import random
    import json
    import numpy as np
    import networkx as nx
    from networkx.readwrite import json_graph
except ImportError:
    msg = "You are missing a library that is required to run this example test-case."
    raise ImportError(msg)

def random_linear_comb(n_mon_tot, max_mon_side, p_side_chain):
    """
    Parameters
    ----------
    n_mon_tot: int
        total number of monomers in the polymer
    max_mon_side:
        the maximum number of monomers per branched off arm
    p_side_chain: list[float]
        probabilities to add at a level corrsponding to the
        position in the list i.e. probability of adding a monomer
        to a side chain beyond the back-bone is list[0], adding a monomer branched
        off 2 level from the back-bone is list[1].

    Returns
    --------
    nx.Graph
    """
    G = nx.Graph()
    # add the first edge to get started
    G.add_edge(0, 1, **{"linktype": "a16"})
    # backbone-nodes
    bb_nodes = [0, 1]
    ndx = 1
    prev_node = "BB"
    while ndx < n_mon_tot:
        # then we make the choice back-bone or side-chain
        # but only if we did not add a side-chain before
        # because we build a comb which has only 1 side chain
        # per back-bone otherwise it's a tree
        if prev_node == "BB":
            choice = random.choices(["BB", "SC"], weights=[1-p_side_chain[0], p_side_chain[0]])[0]
        else:
            choice = "BB"

        # if choice is BB we append to the list of BB nodes and continue
        if choice == "BB":
            G.add_edge(bb_nodes[-1], ndx+1, **{"linktype": "a16"})
            ndx += 1
            bb_nodes.append(ndx)
            prev_node = "BB"

        # if choice is SC we start growing a arm
        else:
            # add the first node of the arm
            G.add_edge(bb_nodes[-1], ndx+1, **{"linktype": "a13"})
            ndx += 1
            prev_node = "SC"
            # an arm can have at most max_side_chains monomers beyond first level
            # we add them until we draw a stop
            for sc_idxs in range(1, max_mon_side):
                choice = random.choices(["stop", "add"], weights=[
                                        1-p_side_chain[sc_idxs], p_side_chain[sc_idxs]])[0]
                if choice == "stop":
                    break
                else:
                    G.add_edge(ndx, ndx+1, **{"linktype": "a16"})
                    ndx += 1
                prev_node = "SC"
                # if we exceed the max number of monomers we terminate
                if ndx >= n_mon_tot:
                    return G

        # if we exceed the max number of monomers we terminate
        if ndx >= n_mon_tot:
            return G


def probability_by_degree_of_polymerization(p, N):
    """
    Computes the probability for a chain of degree of polymerization
    N based on the extend of reaction p for linear condensation reactions.
    See Colby and Rubinstein Polymer Physics.
    """
    return N*p**(N-1)*(1-p)**2.


def __main__():
    # input arguments
    handle = sys.argv[1]
    nmols = sys.argv[2]


    weights = np.arange(2, 600)
    weight_distribution = probability_by_degree_of_polymerization(0.970, weights)
    # select 500 chains from the molecular weight distribution
    random_sample = np.random.choice(weights, p=weight_distribution/np.sum(weight_distribution),
                                     size=int(nmols))

    for idx, n_mon in enumerate(random_sample):
        graph = random_linear_comb(n_mon, 2, [0.05, 0.95, 0.95, 0.5])
        nx.set_node_attributes(graph, "DEX", "resname")

        resids = {node: node+1 for node in graph.nodes}
        nx.set_node_attributes(graph, resids, "resid")

        g_json = json_graph.node_link_data(graph)
        filename = handle + str(n_mon) + "_" + str(idx) + ".json"
        with open(filename, "w") as file_handle:
            json.dump(g_json, file_handle, indent=2)

__main__()
