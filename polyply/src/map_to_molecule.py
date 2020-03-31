from polyply.src.processor import Processor

class MapToMolecule(Processor):
    """
    This processor takes a :class:`MetaMolecule` and generates a
    :class:`vermouth.molecule.Molecule`, which consists at this stage
    of disconnected blocks. These blocks can be connected using the
    :class:`ConnectMolecule` processor. It can either run on a
    single meta molecule or a system. The later is currently not
    implemented.
    """

    @staticmethod
    def add_blocks(meta_molecule):
        force_field = meta_molecule.force_field
        name = meta_molecule.nodes[0]["resname"]
        new_mol = force_field.blocks[name].to_molecule()

        for node in list(meta_molecule.nodes.keys())[1:]:
            resname = meta_molecule.nodes[node]["resname"]
            new_mol.merge_molecule(force_field.blocks[resname])

        return new_mol

    def run_molecule(self, meta_molecule):
        """
        Process a single molecule. Must be implemented by subclasses.
        Parameters
        ----------
        molecule: polyply.src.meta_molecule.MetaMolecule
             The meta molecule to process.
        Returns
        -------
        vermouth.molecule.Molecule
            Either the provided molecule, or a brand new one.
        """
        new_molecule = self.add_blocks(meta_molecule)
        meta_molecule.molecule = new_molecule
        return meta_molecule
