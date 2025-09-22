import numpy as np

from vermouth.molecule import Molecule

OXDNA_TO_NM = 0.8518

def read_oxdna(file_name, exclude=(), strand_lengths=None):
    """
    Parse an oxDNA configuration file to create a molecule.

    Parameters
    ----------
    file_name: str
        The file to read.
    exclude: collections.abc.Container[str]
        Atoms that have one of these residue names will not be included.

    strand_lengths: list of int or None
        The lengths of each DNA strand. If None, the function will attempt to
        guess based on the total number of nucleotides (assuming equal strands
        for even numbers). For example, [12, 12] would indicate two strands of
        12 nucleotides each.

    Returns
    -------
    vermouth.molecule.Molecule
        The parsed molecule. Will not contain edges.
    """
    molecule = Molecule()
    nucleotides = []

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
            major_groove_vector /= np.linalg.norm(major_groove_vector)

            # Store the orientation frame
            orientation_frame = np.column_stack([base_vector, major_groove_vector, base_normal_vector])

            properties['orientation'] = orientation_frame

            # Add to nucleotides list
            nucleotides.append(properties)

    # If strand_lengths is not provided, try to guess based on the number of nucleotides
    if strand_lengths is None:
        # If the number of nucleotides is even, assume it's a duplex with equal strand lengths
        if len(nucleotides) % 2 == 0:
            strand_lengths = [len(nucleotides) // 2, len(nucleotides) // 2]
        else:
            raise IOError("We couldn't infer the duplex strands in the system.")

    # Process nucleotides based on strand_lengths and convert 3' to 5' to 5' to 3'
    processed_nucleotides = []

    # Process each strand separately
    start_idx = 0
    for length in strand_lengths:
        end_idx = start_idx + length
        strand = nucleotides[start_idx:end_idx]

        strand = list(reversed(strand))

        processed_nucleotides.extend(strand)
        start_idx = end_idx

    # Add nucleotides to the molecule
    for idx, properties in enumerate(processed_nucleotides):
        # Set resid
        properties['resid'] = idx + 1

        # Add node to molecule
        molecule.add_node(idx, **properties)

    # Set box
    molecule.box = box

    return molecule
