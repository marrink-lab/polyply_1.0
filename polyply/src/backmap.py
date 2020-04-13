from .processor import Processor
import subprocess

class Backmap(Processor):
    """

    """
    backward_options = [
    "-f",          # Input  GRO/PDB structure
    "-o",          # Output GRO/PDB structure
    "-raw",        # Projected structure before geometric modifications
    "-n",          # Output NDX index file with default groups
    "-p",          # Atomistic target topology
    "-po",         # Output target topology with matching molecules list
    "-pp",         # Processed target topology, with resolved #includes
    "-atomlist",   # Atomlist according to target topology
    "-fc",         # Position restraint force constant
    "-to",         # Output force field
    "-from",       # Input force field
    "-strict",     # Use strict format for PDB files
    "-nt",         # Use neutral termini for proteins
    "-sol",        # Write water
    "-solname",    # Residue name for solvent molecules
    "-kick",       # Random kick added to output atom positions
    "-nopbc",      # Don't try to unbreak residues (like when having large residues in a small box)
    "-mapdir", ]   # Directory where to look for the mapping files

    def _prepare_input(self, meta_molecule):
        

    def _run_backwards(self, meta_molecule):

        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()


    def run_molecule(self, meta_molecule):
        self._random_walk(meta_molecule)
        return meta_molecule
