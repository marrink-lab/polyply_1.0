import numpy as np
import networkx as nx

class Template(nx.Graph):
    """
    A class for storing  residue template information.

    Parameters
    ----------
    resname: str
        The name of the residue
    frame: np.ndarray((3,3))
        Stores the internal reference frame of the residue.
    positions: dict
        Dictionary that stores atom names and positions

    Attributes
    ----------
    positions_arr: np.ndarray((n,3))
        Numpy array which stores the n-atom positions.

    """
    def __init__(self, resname, frame=np.eye(3), positions={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resname = resname
        self.frame = frame
        self.positions_arr = np.array([])
        self.positions = positions

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, dictionary):
        self.positions_arr = np.array(list(dictionary.values()))
        self._positions = dictionary
