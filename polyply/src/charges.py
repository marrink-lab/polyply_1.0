import numpy as np
import networkx as nx
import scipy.optimize

def set_charges(block, res_graph, name):
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
    bond_dipo = np.zeros((len(bonds)))
    for kdx, (idx, jdx) in enumerate(bonds.keys()):
        lb = bonds[(idx, jdx)]
        bond_dipo[kdx] = lb*(charges[idx] - charges[jdx])
    return bond_dipo

def _get_bonds(block, topology=None):
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
                        print(params)
                        bonds[(nodes_to_count[idx], nodes_to_count[jdx])] = float(params)
    return bonds

def equalize_charges(block, topology=None, charge=0):
    block.make_edges_from_interaction_type('bonds')
    keys = nx.get_node_attributes(block, 'charge').keys()
    charges = np.array(list(nx.get_node_attributes(block, 'charge').values()))
    if np.isclose(charges.sum(), 0, atol=1*10**-6):
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
                                          options={'ftol': 0.001, 'maxiter': 100})
    balanced_charges = opt_results['x']
    nx.set_node_attributes(block, dict(zip(keys, balanced_charges)), 'charge')
    return block


#def equalize_charges(molecule, target_charge=0):
#    """
#    Make sure that the total charge of molecule is equal to
#    the target charge by substracting the differences split
#    over all atoms.
#
#    Parameters
#    ----------
#    molecule: :class:`vermouth.molecule.Molecule`
#    target_charge: float
#        the charge of the molecule
#
#    Returns
#    -------
#    molecule
#        the molecule with updated charge attribute
#    """
#    total = nx.get_node_attributes(molecule, "charge")
#    diff = (sum(list(total.values())) - target_charge)/len(molecule.nodes)
#    if np.isclose(diff, 0, atol=0.0001):
#        return molecule
#    for node in molecule.nodes:
#        charge = float(molecule.nodes[node]['charge']) - diff
#        molecule.nodes[node]['charge'] = charge
#    total = nx.get_node_attributes(molecule, "charge")
#    return molecule
