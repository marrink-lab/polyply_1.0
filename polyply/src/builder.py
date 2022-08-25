"""
Base class for all polyply builders.
"""
from conditionals import fulfill_geometrical_constraints, checks_milestones,
                         is_restricted, is_not_overlap

class Builder():
    """
    A builder generates coordiantes for a single meta_molecule at the
    one bead per residue resolution. It defines a protocol to generate
    new positions and automatically checks conditionals that define
    extra build options such as geometrical constraints.
    """
    def __init__(nonbond_matrix, extra_conditionals={}):
        self.nonbond_matrix = nonbond_matrix
        self.conditionals = [fulfill_geometrical_constraints,
                             checks_milestones,
                             is_restricted,
                             is_overlap]
        self.conditionals.append(extra_conditionals)

    def _check_conditionals(new_point, molecule, node):
        """
        Checks that the new point fullfills every conditon
        as dictated by the conditional functions.
        """
        for conditional in self.conditionals:
            if not conditional(new_point, molecule, node):
                return False
        return True

    def remove_positions(mol_idx, nodes):
        self.nonbond_matrix.remove_positions(self.mol_idx, nodes)

    def add_position(new_point, mol_idx, node, start=False):
        """
        If conditionals are fullfilled for the node then
        add the position to the nonbond_matrix.
        """
        if self._check_conditionals(new_point, molecule.nodes[node]):
            self.nonbond_matrix.add_positions(new_point,
                                              mol_idx,
                                              node,
                                              start=start)

    def add_positions(new_points, mol_idx, node, start=False):
        """
        Wrapper around add_position, for looping about multiple
        positions.
        """
        for point in new_points:
             self.nonbond_matrix.add_positions(new_point,
                                              mol_idx,
                                              node,
                                              start=start)
    def build_protocol():
        """
        Must be defined in subclass of builder call update
        positions on every new_point added.
        """
        pass

    def run_molecule(molecule, starting_point):
        self.build_protocol(molecule, starting_point)
        return molecule
