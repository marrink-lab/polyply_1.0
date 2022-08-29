"""
Base class for all polyply builders.
"""
from .meta_molecule import _find_starting_node
from .conditionals import CONDITIONALS

class MaxIterationError(Exception):
    """
    Raised when the maximum number of iterations
    is reached in a building attempt.
    """

class Builder():
    """
    A builder generates coordiantes for a single meta_molecule at the
    one bead per residue resolution. It defines a protocol to generate
    new positions and automatically checks conditionals that define
    extra build options such as geometrical constraints.
    """
    def __init__(self,
                 nonbond_matrix,
                 maxiter):
        """
        Paramters
        ----------
        nonbond_matrix: :class:`polyply.src.nonbond_engine.NonBondMatrix`
        maxiter: int
            number of tries to build a coordiante before returning

        Attributes
        ----------
        mol_idx: index of the molecule in the topology list of molecules
        molecule: molecule: :class:`polyply.Molecule`
            meta_molecule for which the point should be added
        """
        self.nonbond_matrix = nonbond_matrix
        self.maxiter = maxiter

        # convience attributes
        self.mol_idx = None
        self.molecule = None
        self.step_count = 0
        self.path = None

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

    def add_position(self, new_point, node, start=False):
        """
        If conditionals are fullfilled for the node then
        add the position to the nonbond_matrix.

        Parameters
        ----------
        new_point: numpy.ndarray
            the coordinates of the point to add
        nodes: list[abc.hashable]
            list of nodes from which to remove the positions
        start: bool
            if True triggers rebuilding of the positions tree

        Returns
        -------
        bool
            is True if the position could be added
        """
        if self.step_count > self.maxiter:
            raise MaxIterationError()

        if self._check_conditionals(new_point, self.molecule, node):
            self.nonbond_matrix.add_positions(new_point,
                                              self.mol_idx,
                                              node,
                                              start=start)
            self.step_count = 0
            return True
        else:
            self.step_count += 1
            return False

    def add_positions(self, new_points, nodes):
        """
        Add positions of multiple nodes if each node full-fills the conditionals.
        If not then none of the points is added and the False is returned.

        Parameters
        ----------
        new_point: numpy.ndarray
            the coordinates of the point to add
        nodes: list[abc.hashable]
            list of nodes from which to remove the positions

        Returns
        -------
        bool
            is True if the position could be added
        """
        if self.step_count > self.maxiter:
            raise MaxIterationError()

        for node, point in zip(nodes, new_points):
            if not self._check_conditionals(point, self.molecule, node):
                self.step_count += 1
                return False

        for node, point in zip(nodes, new_points):
            self.nonbond_matrix.add_positions(point,
                                              self.mol_idx,
                                              node,
                                              start=False)
        self.step_count = 0
        return True

    def build_protocol(self):
        """
        Must be defined in subclass of builder call update
        positions on every new_point added. In addition
        this function must return a bool indicating if the
        position building has worked or failed.
        """
        return True

    def execute_build_protocol(self):
        """
        This wraps the build protocol method and handles maximum
        iteration counts.
        """
        try:
            status = self.build_protocol()
        except MaxIterationError:
            return False
        return status

    def _prepare_build(self, starting_point, starting_node):
        """
        Called before the building stage. This function finds
        the starting node and adds the first coordinate. It also
        resets all counters associated with a build cycle (i.e. a
        call to the run_molecule method).
        """
        if starting_node:
            first_node = _find_starting_node(self.molecule)
        else:
            first_node = starting_node

        self.molecule.root = first_node

        if "position" not in self.molecule.nodes[first_node]:
            prepare_status = self.add_position(starting_point, first_node, start=True)
        else:
            prepare_status = True

        self.path = list(self.molecule.search_tree.edges)
        self.step_count = 0
        return prepare_status

    def run_molecule(self, molecule, mol_idx, starting_point, starting_node=None):
        """
        Overwrites run_molecule of the Processor class and
        adds the alternative starting_point argument.

        Parameters
        ----------
        molecule: :class:`polyply.MetaMolecule`
            meta_molecule for which the point should be added
        mol_idx: int
            index of the molecule in topology list
        sarting_point: numpy.ndarray
            the coordinates of the point to add
        starting_node: abc.hashable
            first node to start building at; if default None
            first node without defined coordinates is taken

        Returns
        -------
        bool
            if the building process has completed successfully
        """
        # update some class variables for convience
        self.molecule = molecule
        self.mol_idx = mol_idx
        is_prepared = self._prepare_build(starting_point, starting_node)
        if not is_prepared:
            return False

        build_status = self.execute_build_protocol()
        return build_status
