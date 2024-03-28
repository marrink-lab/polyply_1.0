import numpy as np
import networkx as nx
import scipy.optimize

def set_charges(block, res_graph, name):
    """
    Set the charges of `block` by finding the most central
    residue in res_graph that matches the residue `name` of
    block.

    Parameters
    ----------
    block: :class:`vermouth.molecule.Block`
        block describing single residue
    res_graph: nx.Graph
        residue graph
    name: str
        residue name

    Returns
    -------
    :class:`vermouth.molecule.Block`
        the block with updated charges
    """
    resnames = nx.get_node_attributes(res_graph, 'resname')
    centrality = nx.betweenness_centrality(res_graph)
    score = -1
    most_central_node = None
    for node, resname in resnames.items():
        if resname == name and centrality[node] > score:
            score = centrality[node]
            most_central_node = node
    charges_tmp = nx.get_node_attributes(res_graph.nodes[most_central_node]['graph'], 'charge')
    atomnames = nx.get_node_attributes(res_graph.nodes[most_central_node]['graph'], 'atomname')
    charges = {atomname: charges_tmp[node] for node, atomname in atomnames.items()}
    for node in block.nodes:
        block.nodes[node]['charge'] = charges[block.nodes[node]['atomname']]
    return block

def bond_dipoles(bonds, charges):
    """
    Compute bond dipole moments from charges
    and bondlengths. The charges array must
    match the numeric bond dict keys.

    Parameters
    ----------
    bonds: dict[tuple(int, int)][float]
        the bond length indexed by atom indices
    charges: np.array
        array of charges

    Returns
    -------
    np.array
        the bond dipoles
    """
    bond_dipo = np.zeros((len(bonds)))
    for kdx, (idx, jdx) in enumerate(bonds.keys()):
        lb = bonds[(idx, jdx)]
        bond_dipo[kdx] = lb*(charges[idx] - charges[jdx])
    return bond_dipo

def _get_bonds(block, topology=None):
    """
    Extract a bond length dict from block. If topology
    is given bond lengths may be looked up by type.

    Parameters
    ----------
    block: :class:`vermouth.molecule.Block`
    topology: :class:`polyply.src.topology.Topology`

    Returns
    -------
    dict
        a dict of edges and their bond length
    """
    bonds = {}
    atoms = block.nodes
    nodes_to_count = {node: count for count, node in enumerate(block.nodes)}
    for idx, jdx in block.edges:
        for bond in block.interactions['bonds']:
            if tuple(bond.atoms) in [(idx, jdx), (jdx, idx)]:
                try:
                    bonds[(nodes_to_count[idx], nodes_to_count[jdx])] = float(bond.parameters[1])
                except IndexError:
                    if topology:
                        batoms = (atoms[idx]['atype'],
                                  atoms[jdx]['atype'])
                        if batoms in topology.types['bonds']:
                            params = topology.types['bonds'][batoms][0][0][1]
                        elif batoms[::-1] in topology.types['bonds']:
                            params = topology.types['bonds'][batoms[::-1]][0][0][1]
                        bonds[(nodes_to_count[idx], nodes_to_count[jdx])] = float(params)
                    else:
                        msg = ("Cannot find bond lengths. If your force field uses bondtypes lile"
                               "Charmm you need to provide a topology file.")
                        raise ValueError(msg)
    return bonds

def balance_charges(block, charge=0, tol=10**-8, decimals=8, topology=None):
    """
    Given a block and a total charge for that block
    balance the charge until the total charge of the
    block is exactly the same as set. The balancing
    takes also into account to retain the bond dipole
    moments as closely as possible such that ideally
    the electrostatics are as little influenced as
    possible due to rescaling. A topology is only
    needed if the force field uses bondtypes.

    Parameters
    ----------
    block: :class:`vermouth.molecule.Block`
    topology: :class:`polyply.src.topology.Topology`
    charge: float
        total charge of the residue

    Returns
    -------
    :class:`vermouth.molecule.Block`
        block with updated charges
    """
    if len(block.nodes) < 2:
        return block

    block.make_edges_from_interaction_type('bonds')
    keys = nx.get_node_attributes(block, 'charge').keys()
    charges = np.array(list(nx.get_node_attributes(block, 'charge').values()))
    if np.isclose(charges.sum(), 0, atol=tol):
        return block

    # we need to equalize the charge
    bonds = _get_bonds(block, topology)
    ref_dipoles = bond_dipoles(bonds, charges)

    # the loss consists of the deviation of the
    # sum of charges from zero and the difference
    # in the original bond dipole moments
    def loss(arr):
        arr.reshape(-1)
        curr_dipoles = bond_dipoles(bonds, arr)
        crg_dev = np.abs(charge - arr.sum())
        loss = crg_dev + np.sum(np.square(ref_dipoles -  curr_dipoles))
        return loss

    opt_results = scipy.optimize.minimize(loss, charges, method='L-BFGS-B',
                                          options={'ftol': tol, 'maxiter': 100})
    balanced_charges = np.around(opt_results['x'], decimals)
    nx.set_node_attributes(block, dict(zip(keys, balanced_charges)), 'charge')
    return block
