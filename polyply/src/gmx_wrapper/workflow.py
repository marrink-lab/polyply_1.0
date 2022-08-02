"""
Useful classes for integrating workflows with gromacs wrapper.
"""
import os
from pathlib import Path
import vermouth
import gromacs as gmx
from gromacs.run import MDrunner


class TopologyGMXRunner(MDrunner):
    """
    Run energy minization on a polyply topology.
    To use this method instantiate the class and
    then run the simulation with run() or run_check().
    The latter returns a success status of the simulation.
    After the simulation the coordinates in the topolgy
    object will be updated in place.
    """
    def __init__(self,
                 topology,
                 toppath,
                 mdppath,
                 dirname=os.path.curdir,
                 **kwargs):
        """
        Set some enviroment variables for the run.
        """
        self.topology = topology
        self.toppath = toppath
        self.mdppath = mdppath
        self.startpath = dirname / Path("start.gro")
        self.outpath = dirname / Path(kwargs.get('deffnm', 'confout') + ".gro")
        self.tprname = dirname / Path(kwargs.get('deffnm', 'topol') + ".tpr")
        kwargs['s'] = self.tprname
        super().__init__(dirname=dirname, **kwargs)

    def prehook(self, **kwargs):
        """
        Write the coordinates of the current system and grompp
        the input parameters.
        """
        # write the system coordinates to file
        system = self.topology.convert_to_vermouth_system()
        vermouth.gmx.gro.write_gro(system, self.startpath, precision=7,
                                   title="starting coordinates",
                                   box=self.topology.box, defer_writing=False)
        # grompp everything in the directory
        gmx.grompp(f=self.mdppath,
                   c=self.startpath,
                   p=self.toppath,
                   o=self.tprname)

    def posthook(self, **kwargs):
        """
        Read in the new coordinates from the completed run.
        """
        self.topology.add_positions_from_file(self.outpath)
