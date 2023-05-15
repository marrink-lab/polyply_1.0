import numpy as np
from .gen_moving_frame import ConstructFrame

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

    rotation_per_bp = .34
    construct_frame = ConstructFrame(rotation_per_bp=rotation_per_bp)
    moving_frame = construct_frame.run(meta_molecule)

    # Write the template as array
    template_arr = np.zeros((3, len(template)))
    key_to_ndx = {}
    for ndx, key in enumerate(template.keys()):
        template_arr[:, ndx] = template[key]
        key_to_ndx[key] = ndx

    template_rotated_arr = moving_frame[current_node] @ template_arr

    # Write the template back as dictionary
    template_rotated = {}
    for key, ndx in key_to_ndx.items():
        template_rotated[key] = template_rotated_arr[:, ndx]

    return template_rotated
