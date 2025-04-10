import numpy as np

from vermouth.molecule import Molecule

OXDNA_TO_NM = 0.8518

def read_oxdna(file_name, exclude=()):
    """
    Parse an oxDNA configuration file to create a molecule.

    Parameters
    ----------
    file_name: str
        The file to read.
    exclude: collections.abc.Container[str]
        Atoms that have one of these residue names will not be included.

    Returns
    -------
    vermouth.molecule.Molecule
        The parsed molecule. Will not contain edges.
    """
    molecule = Molecule()
    idx = 0

    with open(str(file_name)) as oxdna:
        # Parse header
        next(oxdna)  # Skip time line

        box_line = next(oxdna)
        box = np.array([float(x) for x in box_line.split('=')[1].strip().split()])
        box *= OXDNA_TO_NM

        next(oxdna)  # Skip energy line

        # Parse nucleotide data
        for line in oxdna:
            values = line.strip().split()
            if len(values) < 9:  # We need at least position and orientation
                continue

            properties = {}

            # Position
            position = np.array([float(values[0]), float(values[1]), float(values[2])], dtype=float)
            position *= OXDNA_TO_NM
            properties['position'] = position

            # Base vector
            base_vector = np.array([float(values[3]), float(values[4]), float(values[5])], dtype=float)
            base_vector /= np.linalg.norm(base_vector)

            # Base normal vector
            base_normal_vector = np.array([float(values[6]), float(values[7]), float(values[8])], dtype=float)
            base_normal_vector /= np.linalg.norm(base_normal_vector)

            # Construct major groove vector
            major_groove_vector = np.cross(base_normal_vector, base_vector)
            # major_groove_vector = np.cross(base_vector, base_normal_vector)

            # Store the orientation frame
            # orientation_frame = np.column_stack([base_vector, major_groove_vector, base_normal_vector])
            orientation_frame = np.column_stack([major_groove_vector, -base_vector, base_normal_vector])
            properties['orientation'] = orientation_frame

            # Default values
            properties['resid'] = idx + 1

            molecule.add_node(idx, **properties)
            idx += 1

    # Set box
    molecule.box = box

    return molecule
