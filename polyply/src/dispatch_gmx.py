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

import subprocess
import vermouth
import os

class GMXError(Exception):
    """
    Class for handeling GROMACS errors. This class only
    prints the essentail part to the error message when
    raised.
    """

    def __init__(self, stderr):
        lines = stderr.decode("utf-8")
        message = "GROMACS terminated with the following error:\n"
        message = message + lines
        super().__init__(message)


class DispatchGMX:
    """
    Abstract base class for running gromacs from
    within polyply.
    """

    def __init__(self, gmx_exe, workdir, gmx_path):
        self.gmx_exe = gmx_exe
        self.workdir = workdir
        self.gmx_path = gmx_path

    def run(self, command_args, stdin=None):
        """
        Run GMX command with parameters.
        """
        os.chdir(self.workdir)
        command = [self.gmx_path, self.gmx_exe]

        for token, value in command_args.items():
            command.append("-" + token)
            command.append(value)

        output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, input=stdin)
        # ToDo record notes and warnings perhaps with a log file
        if output.returncode == 1:
            raise GMXError(output.stderr)

        return output.stdout.decode('utf-8'), output.stderr.decode('utf-8')
