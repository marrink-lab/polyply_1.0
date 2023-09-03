import numpy as np

def orient_by_frames(meta_molecule, current_node, template, built_nodes):
    """
    Given a `template` and a `node` of a `meta_molecule` at lower resolution
    align the template to a constructed rotation minimizing frame.

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

    # This is a temporary solution, it looks a bit silly. In the end the template will be
    # stored in each meta_molecule node.
    base_vector =  meta_molecule.nodes[current_node]["base vector"]
    base_normal_vector = meta_molecule.nodes[current_node]["base normal vector"]
    base_binormal_vector = np.cross(base_normal_vector, base_vector)
    template.frame = np.array([base_vector, base_binormal_vector, base_normal_vector]).T

    # Orient template to frame by tranforming each position vector
    template_rotated_arr =  np.einsum('ij, kj -> ki', template.frame, template.positions_arr)

    # Write the rotated template back as dictionary
    template_rotated = {}
    for ndx, key in enumerate(template.positions):
        template_rotated[key] = template_rotated_arr[ndx]

    return template_rotated
