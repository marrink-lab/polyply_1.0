# Copyright 2020 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import subprocess
from pathlib import Path
import tempfile
import vermouth
from .dispatch_gmx import DispatchGMX


def _find_max_force(mdrun_out):
    for line in mdrun_out.split("\n"):
        if "Maximum force" in line:
            force = line.split()[3]
    return float(force)

class GMXMinimize():
    """
    Processor for energy minimizing a system
    """

    def __init__(self, top_file, mdp_file, max_force=10**3.0, max_cycles=10):
        self.top_file = Path(top_file).absolute()
        self.mdp_file = Path(mdp_file).absolute()
        self.max_cycles = max_cycles
        self.max_force = max_force
        self.top_dir = os.getcwd()

    def run_topology(self, topology):

        tmpdir = tempfile.TemporaryDirectory(dir=self.top_dir)
        mdrun = DispatchGMX("mdrun", tmpdir.name, "gmx")
        grompp = DispatchGMX("grompp", tmpdir.name, "gmx")

        os.chdir(tmpdir.name)

        system = topology.convert_to_vermouth_system()
        vermouth.gmx.gro.write_gro(system, "start.gro", precision=7,
                                   title='polyply structure', box=(10, 10, 10))

        prev_step = Path("start.gro").absolute()

        for cycle in range(0, self.max_cycles):
            _ = grompp.run({"p":self.top_file, "f": self.mdp_file, "c": prev_step, "o": str(cycle)+".tpr", "maxwarn": "1"})
            mdrun_out, mdrun_err = mdrun.run({"nsteps": "100", "s":  str(cycle)+".tpr", "deffnm":str(cycle)})
            force = _find_max_force(mdrun_err)
            if force < self.max_force:
                break
            prev_step = Path(str(cycle) + ".gro").absolute()
        else:
            tmpdir.cleanup()
            raise IOError

        trjconv = DispatchGMX("trjconv", tmpdir.name, "gmx")
        stdout, stderr = trjconv.run({"s": str(cycle)+".tpr", "o":"final.gro", "pbc": "whole", "f": str(cycle)+".gro"}, stdin=b"System")

        outfile = "final.gro"
        topology.add_positions_from_file(outfile)
        os.chdir(self.top_dir)
        tmpdir.cleanup()
        return topology
