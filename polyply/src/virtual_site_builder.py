import numpy as np
from numpy.linalg import norm
"""
Rules for constructing virtual sites according
to GROMACS definitions.
"""

def vsn(interaction, positions):
    """
    Construct a virtual-siten of type
    1,2,3 from interaction and positions.

    Paratmers
    ---------
    interaction
    positions

    Returns
    -------
    np.ndarray(3)
    """
    weights = np.full((len(interaction.atoms[1:])), 1.)
    coord = np.array([0., 0., 0.])
    for atom, weight in zip(interaction.atoms[1:], weights):
        coord += positions[atom] * weight /np.sum(weights)

    return coord

def vs2(interaction, positions):
    """
    Constructs virtual-site 2.

    Paratmers
    ---------
    interaction
    positions

    Returns
    -------
    np.ndarray(3)
    """
    a = float(interaction.parameters[1])
    wi = 1 - a
    wj = a
    weights = [wi, wj]
    coord = np.array([0., 0., 0.])
    for atom, weight in zip(interaction.atoms[1:], weights):
        coord += positions[atom] * weight
    return coord

def vs3(interaction, positions):
    """
    Constructs virtual-site 3.

    Paratmers
    ---------
    interaction
    positions

    Returns
    -------
    np.ndarray(3)
    """
    a = float(interaction.parameters[1])
    b = float(interaction.parameters[2])
    wi = 1 - a - b
    wj = a
    wk = b
    weights = [wi, wj, wk]
    coord = np.array([0., 0., 0.])
    for atom, weight in zip(interaction.atoms[1:], weights):
        coord += positions[atom] * weight
    return coord

def vs3fd(interaction, positions):
    """
    Construct virtual-site 3fd.

    Paratmers
    ---------
    interaction
    positions

    Returns
    -------
    np.ndarray(3)
    """
    atoms = interaction.atoms[1:]
    r_i, r_j, r_k = positions[atoms[0]], positions[atoms[1]], positions[atoms[2]]
    r_ij = r_j - r_i
    r_jk = r_k - r_j
    r_ij = r_j - r_i
    a = float(interaction.parameters[1])
    b = float(interaction.parameters[2])

    return r_i + (b * (r_ij + a * r_jk)/norm(r_ij + a * r_jk))

def vs3fad(interaction, positions):
    """
    Construct virtual-site 3fad.

    Paratmers
    ---------
    interaction
    positions

    Returns
    -------
    np.ndarray(3)
    """
    atoms = interaction.atoms[1:]
    r_i, r_j, r_k = positions[atoms[0]], positions[atoms[1]], positions[atoms[2]]
    theta, d = float(interaction.parameters[1]), float(interaction.parameters[2])
    r_ij = r_j - r_i
    r_jk = r_k - r_j
    r_ij = r_j - r_i
    r_normal = r_jk - r_ij * np.dot(r_ij, r_jk)/np.dot(r_ij, r_ij)
    angle1 = np.cos(np.deg2rad(theta)) * r_ij/norm(r_ij)
    angle2 = np.sin(np.deg2rad(theta)) * r_normal/norm(r_normal)
    vs = r_i + d * angle1 + d * angle2
    return vs

def vs3out(interaction, positions):
    """
    Construct virtual-site 3out.

    Paratmers
    ---------
    interaction
    positions

    Returns
    -------
    np.ndarray(3)
    """
    atoms = interaction.atoms[1:]
    r_i, r_j, r_k = positions[atoms[0]], positions[atoms[1]], positions[atoms[2]]
    a = float(interaction.parameters[1])
    b = float(interaction.parameters[2])
    c = float(interaction.parameters[3])
    r_ij = r_j - r_i
    r_ik = r_k - r_i
    r_ij = r_j - r_i
    vs = r_i + float(a) * r_ij + float(b) * r_ik + float(c) * np.cross(r_ij, r_ik)
    return vs

def vs4fdn(interaction, positions):
    """
    Construct virtual-site 4fdn.

    Paratmers
    ---------
    interaction
    positions

    Returns
    -------
    np.ndarray(3)
    """
    atoms = interaction.atoms[1:]
    a = float(interaction.parameters[1])
    b = float(interaction.parameters[2])
    c = float(interaction.parameters[3])
    r_i, r_j = positions[atoms[0]], positions[atoms[1]]
    r_k, r_l = positions[atoms[2]], positions[atoms[3]]
    r_ij = r_j - r_i
    r_ik = r_k - r_i
    r_il = r_l - r_i
    r_ja = a * r_ik - r_ij
    r_jb = b * r_il - r_ij
    rm = np.cross(r_ja, r_jb)
    vs = r_i + c * rm/norm(rm)
    return vs

def vserr(interaction, positions):
    raise IOError("Virtual-site 4 with function-type"
                  "1 is deprecated and cannot be used.")

# we don't differentiate between COM, COG
# for virtual_sitesn because the error
# is accaptable in terms of approximate
# structure generation

VIRTUAL_SITES = {('virtual_sitesn', '1'): vsn,
                 ('virtual_sitesn', '2'): vsn,
                 ('virtual_sitesn', '3'): vsn,
                 ('virtual_sites4', '2'): vs4fdn,
                 ('virtual_sites4', '1'): vserr,
                 ('virtual_sites3', '4'): vs3out,
                 ('virtual_sites3', '3'): vs3fad,
                 ('virtual_sites3', '2'): vs3fd,
                 ('virtual_sites3', '1'): vs3,
                 ('virtual_sites2', '1'): vs2
                 }
def construct_vs(vs_type, interaction, positions):
    """
    Construct a vs from atoms in `positions` according
    to `vs_type` and atoms in `interaction`.
    """
    func = interaction.parameters[0]
    return VIRTUAL_SITES[(vs_type, func)](interaction, positions)
