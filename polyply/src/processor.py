import vermouth
import polyply.src

class Processor:
    """
    An abstract base class for processors. Subclasses must implement a
    `run_molecule` method.
    """
    def run_system(self, system):
        """
        Process `system`.
        Parameters
        ----------
        system: vermouth.system.System
            The system to process. Is modified in-place.
        """
        mols = []
        for molecule in system.molecules:
            mols.append(self.run_molecule(molecule))
        system.molecules = mols

    def run_molecule(self, meta_molecule):
        """
        Process a single molecule. Must be implemented by subclasses.
        Parameters
        ----------
        molecule: vermouth.molecule.Molecule
            The molecule to process.
        Returns
        -------
        vermouth.molecule.Molecule
            Either the provided molecule, or a brand new one.
        """
        raise NotImplementedError

