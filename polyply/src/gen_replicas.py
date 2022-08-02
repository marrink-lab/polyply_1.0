"""
Generate multiple system replicas using gen_coords. In essence
this is a wrapper for gen_coords which also makes use of the
gromacs wrapper libary.
"""
import os
from pathlib import Path
from vermouth.log_helpers import StyleAdapter, get_logger
from polyply import gen_coords
from polyply.src.gmx_wrapper.workflow import GMXRunner

LOGGER = StyleAdapter(get_logger(__name__))

def gen_replicas(nrepl,
                 mdppath,
                 logpath,
                 workdir,
                 timeout,
                 **kwargs):
    """
    Generate multiple coordinate replicas of a system.
    this function is a wrapper around gen_coords, which
    also has some options for running simulations.

    Paremeters
    ----------
    nrepl: int
        number of replicas to generate
    mdppath: Path
        file path to mdp-file
    logpath: Path
        file path to log-file
    workdir: Path
        workdir to use
    timeout: float
        time after which gen_coords is timed out
    """
    mdppath = mdppath.resolve()
    workdir = workdir.resolve()
    logpath = logpath.resolve()
    kwargs["toppath"] = kwargs["toppath"].resolve()
    base_name = kwargs['outpath'].name
    os.chdir(workdir)
    for idx in range(0, nrepl):
        os.mkdir(workdir / Path(str(idx)))
        os.chdir(workdir / Path(str(idx)))
        kwargs['outpath'] = workdir / Path(str(idx)) / Path(base_name)
        gen_coords(**kwargs)
        # Running energy minization
        LOGGER.info("running energy minization")
        sim_runner = GMXRunner(kwargs["outpath"],
                               kwargs["toppath"],
                               mdppath,
                               deffnm="min")
        status = sim_runner.run()
        # everything has worked out yay!
        with open(logpath, "a") as logfile:
            if status != 0:
                logfile.write(f"replica {idx} has failed with unkown gmx issue\n")
            if not sim_runner.success:
                logfile.write(f"replica {idx} has failed with infinite forces\n")

            logfile.write(f"replica {idx} was successfully generated\n")

        os.chdir(workdir)
