"""
Base class for all polyply builders.
"""
from .processor import Processor
from .conditionals import CONDITIONALS

class Builder():
    """
    A builder generates coordiantes for a single meta_molecule at the
    one bead per residue resolution. It defines a protocol to generate
    new positions and automatically checks conditionals that define
    extra build options such as geometrical constraints.
    """
    def __init__(self, nonbond_matrix, starting_point=None):
        """
        Attributes
        ----------
        nonbond_matrix: :class:`polyply.src.nonbond_engine.NonBondMatrix`
        starting_point: np.ndarray[3, 1]
        """
        self.nonbond_matrix = nonbond_matrix
        self.starting_point = starting_point

    @staticmethod
    def _check_conditionals(new_point, molecule, node):
        """
        Checks that the new point fullfills every conditon
        as dictated by the conditional functions.
        """
        for conditional in CONDITIONALS:
            if not conditional(new_point, molecule, node):
                return False
        return True

    def remove_positions(self, mol_idx, nodes):
        """
        Remove positions of nodes from the nonbond-matrix by
        molecule index.

        Parameters
        ----------
        mol_idx: int
            the index of the molecule in the system
        nodes: list[abc.hashable]
            list of nodes from which to remove the positions
        """
        self.nonbond_matrix.remove_positions(mol_idx, nodes)

    def add_position(self, new_point, molecule, mol_idx, node, start=False):
        """
        If conditionals are fullfilled for the node then
        add the position to the nonbond_matrix.

        Parameters
        ----------
        new_point: numpy.ndarray
            the coordinates of the point to add
        molecule: :class:`polyply.Molecule`
            meta_molecule for which the point should be added
        mol_idx: int
            the index of the molecule in the system
        nodes: list[abc.hashable]
            list of nodes from which to remove the positions
        start: bool
            if True triggers rebuilding of the positions tree

        Returns
        -------
        bool
            is True if the position could be added
        """
        if self._check_conditionals(new_point, molecule, node):
            self.nonbond_matrix.add_positions(new_point,
                                              mol_idx,
                                              node,
                                              start=start)
            return True
        else:
            return False

    def add_positions(self, new_points, molecule, mol_idx, nodes):
        """
        Add positions of multiple nodes if each node full-fills the conditionals.
        If not then none of the points is added and the False is returned.

        Parameters
        ----------
        new_point: numpy.ndarray
            the coordinates of the point to add
        molecule: :class:`polyply.Molecule`
            meta_molecule for which the point should be added
        mol_idx: int
            the index of the molecule in the system
        nodes: list[abc.hashable]
            list of nodes from which to remove the positions

        Returns
        -------
        bool
            is True if the position could be added
        """
        for node, point in zip(nodes, new_points):
            if not self._check_conditionals(point, molecule, node):
                return False

        for node, point in zip(nodes, new_points):
            self.nonbond_matrix.add_positions(point,
                                              mol_idx,
                                              node,
                                              start=False)
        return True

    def build_protocol(self, molecule):
        """
        Must be defined in subclass of builder call update
        positions on every new_point added.
        """
        pass

    def run_molecule(self, molecule, starting_point=None):
        """
        Overwrites run_molecule of the Processor class and
        adds the alternative starting_point argument.

        Parameters
        ----------
        molecule: :class:`polyply.MetaMolecule`
            meta_molecule for which the point should be added
        sarting_point: numpy.ndarray
            the coordinates of the point to add

        Returns
        -------
        :class:`polyply.MetaMolecule`
        """
        if starting_point:
            self.starting_point = starting_point
        self.build_protocol(molecule)
        return molecule
