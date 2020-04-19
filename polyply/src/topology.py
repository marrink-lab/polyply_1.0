"""
Provides a class used to describe a gromacs topology and all assciated data.
"""

from vermouth.system import System
from vermouth.forcefield import ForceField
from .top_parser import read_topology

class Topology(System):
    """
    Ties together vermoth molecule definitions, and
    Gromacs topology information.

    Parameters
    ----------
    force_field: :class:`vermouth.forcefield.ForceField`
        A force field object.
    name: str, optional
        The name of the topology.

    Attributes
    ----------
    molecules: list[:class:`~vermouth.molecule.Molecule`]
        The molecules in the system.
    force_field: a :class:`vermouth.forcefield.ForceField`
    nonbond_params: dict
        A dictionary of all nonbonded parameters
    types: dict
        A dictionary of all typed parameter
    defines: list
        A list of everything that is defined
    """

    def __init__(self, force_field, name=None):
        self.name = name
        self.molecules = []
        self._force_field = None
        self.force_field = force_field
        self.defaults = {}
        self.defines = {}
        self.discription = []
        self.types = {}
        self.nonbond_params = {}

    @classmethod
    def from_gmx_topfile(cls, path, name):
        """
        Read a gromacs topology file and return an topology object.

        Parameters
        ----------
        path:  str
           The name of the topology file
        name:  str
           The name of the system
        """
        with open(path, 'r') as _file:
            lines = _file.readlines()

        force_field = ForceField(name)
        topology = cls(force_field=force_field, name=name)
        topology = read_topology(lines, topology)
        return topology
