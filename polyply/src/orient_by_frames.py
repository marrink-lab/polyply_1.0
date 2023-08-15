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

    template_rotated_arr = (template.frame @ template.positions_arr.T).T

    # Write the rotated template back as dictionary
    template_rotated = {}
    for ndx, key in enumerate(template.positions):
        template_rotated[key] = template_rotated_arr[ndx]

    return template_rotated
