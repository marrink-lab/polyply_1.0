"""
Useful classes for integrating workflows with gromacs wrapper.
"""
import os
from pathlib import Path
import gromacs as gmx
from gromacs.run import MDrunner
import gromacs.environment
gromacs.environment.flags['capture_output'] = "file"

class GMXRunner(MDrunner):
    """
    Run energy minization on a polyply topology.
    To use this method instantiate the class and
    then run the simulation with run() or run_check().
    The latter returns a success status of the simulation.
    After the simulation the coordinates in the topolgy
    object will be updated in place.
    """
    def __init__(self,
                 coordpath,
                 toppath,
                 mdppath,
                 dirname=os.path.curdir,
                 **kwargs):
        """
        Set some enviroment variables for the run.
        """
        self.coordpath = coordpath
        self.toppath = toppath
        self.mdppath = mdppath
        self.success = False
        self.startpath = dirname / Path("start.gro")
        self.outpath = dirname / Path(kwargs.get('deffnm', 'confout') + ".gro")
        self.tprname = dirname / Path(kwargs.get('deffnm', 'topol') + ".tpr")
        self.deffnm = kwargs.get('deffnm', None)
        kwargs['s'] = self.tprname
        super().__init__(dirname=dirname, **kwargs)

    def prehook(self, **kwargs):
        """
        Write the coordinates of the current system and grompp
        the input parameters.
        """
        # grompp everything in the directory
        gmx.grompp(f=self.mdppath,
                   c=self.coordpath,
                   p=self.toppath,
                   o=self.tprname)

    def posthook(self, **kwargs):
        """
        Make sure that the energy minization did not result into
        infinite forces.
        """
        if self.deffnm:
            logpath = Path(self.deffnm + ".log")
        else:
            logpath = Path("md.log")

        with open(logpath, "r") as logfile:
            for line in logfile.readlines():
                if "Norm" in line:
                    maxforce = line.strip().split()[-1]
                    if maxforce == "inf":
                        self.success = False
                    else:
                        self.success = True
