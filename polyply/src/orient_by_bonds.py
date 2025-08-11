import numpy as np
import scipy.optimize
import networkx as nx
from .linalg_functions import rotate_xyz
from .graph_utils import find_connecting_edges
from .linalg_functions import norm_matrix

def orient_by_bonds(meta_molecule, current_node, template, built_nodes):
    """
    Given a `template` and a `node` of a `meta_molecule` at lower resolution
    find the bonded interactions connecting the higher resolution template
    to its neighbours and orient a template such that the atoms point torwards
    the neighbours. In case some nodes of meta_molecule have already been built
    at the lower resolution they can be provided as `built_nodes`. In case the
    lower resolution is already built the atoms will be oriented torward the lower
    resolution atom participating in the bonded interaction.

    Parameters:
    -----------
    meta_molecule: :class:`polyply.src.meta_molecule`
    current_node:
        node key of the node in meta_molecule to which template referes to
    template: dict[collections.abc.Hashable, np.ndarray]
        dict of positions referring to the lower resolution atoms of node
    built_nodes: list
        list of meta_molecule node keys of residues that are already built

    Returns:
    --------
    dict
        the oriented template
    """
    # 1. find neighbours at meta_mol level
    neighbours = nx.all_neighbors(meta_molecule, current_node)
    current_resid = meta_molecule.nodes[current_node]["resid"]

    # 2. find connecting atoms at low-res level
    edges = []
    ref_nodes = []
    for node in neighbours:
        resid = meta_molecule.nodes[node]["resid"]
        edge = find_connecting_edges(meta_molecule,
                                     meta_molecule.molecule,
                                     (node, current_node))
        edges += edge
        ref_nodes.extend([node]*len(edge))

    # 3. build coordinate system
    ref_coords = np.zeros((3, len(edges)))
    opt_coords = np.zeros((3, len(edges)))

    for ndx, edge in enumerate(edges):
        for atom in edge:
            resid = meta_molecule.molecule.nodes[atom]["resid"]
            if resid == current_resid:
                current_atom = atom
            else:
                ref_atom = atom
                ref_resid = resid

        # the reference residue has already been build so we take the lower
        # resolution coordinates as reference
        if ref_resid in built_nodes:
            atom_name = meta_molecule.molecule.nodes[current_atom]["atomname"]

            # record the coordinates of the atom that is rotated
            opt_coords[:, ndx] = template[atom_name]

            # given the reference atom that already exits translate it to the origin
            # of the rotation, this will be the reference point for rotation
            ref_coords[:, ndx] = meta_molecule.molecule.nodes[ref_atom]["position"] -\
                                 meta_molecule.nodes[current_node]["position"]

        # the reference residue has not been build the CG center is taken as
        # reference
        else:
            atom_name = meta_molecule.molecule.nodes[current_atom]["atomname"]
            cg_node = ref_nodes[ndx] #find_atoms(meta_molecule, "resid", ref_resid)[0]

            # record the coordinates of the atom that is rotated
            opt_coords[:, ndx] = template[atom_name]

            # as the reference atom is not built take the cg node as reference point
            # for rotation; translate it to origin
            ref_coords[:, ndx] = meta_molecule.nodes[cg_node]["position"] -\
                                 meta_molecule.nodes[current_node]["position"]


    # 4. optimize the distance between reference nodes and nodes to be placed
    # only using rotation of the complete template
    #@profile
    def target_function(angles):
        rotated = rotate_xyz(opt_coords, angles[0], angles[1], angles[2])
        diff = rotated - ref_coords
        score = norm_matrix(diff)
        return score

    # choose random starting angles
    angles = np.random.uniform(low=0, high=2*np.pi, size=(3))
    opt_results = scipy.optimize.minimize(target_function, angles, method='L-BFGS-B',
                                          options={'ftol':0.01, 'maxiter': 400})

    # 5. write the template as array and rotate it corrsponding to the result above
    template_arr = np.zeros((3, len(template)))
    key_to_ndx = {}
    for ndx, key in enumerate(template.keys()):
        template_arr[:, ndx] = template[key]
        key_to_ndx[key] = ndx

    angles = opt_results['x']
    template_rotated_arr = rotate_xyz(template_arr, angles[0], angles[1], angles[2])

    # 6. write the template back as dictionary
    template_rotated = {}
    for key, ndx in key_to_ndx.items():
        template_rotated[key] = template_rotated_arr[:, ndx]

    return template_rotated
