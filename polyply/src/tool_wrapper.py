from .processor import Processor

class ToolWrapper():

    def __init__(self, executable, tool_args, *args, **kwargs):
        self.executable = executable
        self.tool_args = tool_args
        self._tool_name = None

    @property
    def tool_name(self):
        pass

    def _execute(self, **kwargs):
        """
        The actual running of the additional tool.
        """
        command = [self.executable]
        for arg, value in self.tool_args.items():
            command.append("-" + "arg")
            command.append(value)

        output = subprocess.run(command)
        return output.stdout.decode('utf-8'), output.stderr.decode('utf-8')

    def _post_process(self):
        """
        This function needs to be defined at subclass level, and
        defines how the coordinates are mapped to the topology.
        """
        pass

    def _perpare(self):
        """
        This function can be defined if anything special has to be
        prepared for the tool like initial coordinates or topology
        files.
        """
        pass

    def run_system(self, topology):
        """
        Called to run the construction based on the tool.
        """
        topology = self._prepare(topology)
        self._execute(topology)
        topology = self._post_process(topolgoy)
        return topology

class FlatMembraneBuilder(Builder, ToolWrapper):

    def __init__(self, executable, *args, **kwargs):
        tool_args = {"str": "input.str",
                     "Bondlength": 0.2,
                     "LLIP": "Martini3.LIN",
                     "defout": "system",
                     "function": "analytical_shape",
                    }
        super().__init__(executable, tool_args, *args, **kwargs)

    @property
    def tool_name(self):
        return "TS2CG"

    def _prepare(self):
        with open("input.str", w) as _file:
            _file.write ...


        
