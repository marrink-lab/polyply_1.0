


class WormLikeCoords(Processor):
    """
    This processor takes a a class:`polyply.src.MetaMolecule` and
    generates coordinates.
    """

    def run_molecule(self, meta_molecule):
        """
        Process a single molecule. Must be implemented by subclasses.

        Parameters
        ----------
        molecule: :class:`polyply.src.meta_molecule.MetaMolecule`
             The meta molecule to process.

        Returns
        ---------
        :class: `polyply.src.meta_molecule.MetaMolecule`
        """

        molecule = meta_molecule.molecule
        force_field = meta_molecule.force_field

        super_coords = random_walk(meta_molecule)
        high_detail_coords = backmap(meta_molecule, super_coords)
      
