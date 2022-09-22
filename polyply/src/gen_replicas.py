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
                 mdppaths,
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
    mdppaths: list[Path]
        file path to mdp-file
    logpath: Path
        file path to log-file
    workdir: Path
        workdir to use
    timeout: float
        time after which gen_coords is timed out
    """
    mdps_resolved = []
    for mdppath in mdppaths:
        mdps_resolved.append(mdppath.resolve())

    workdir = workdir.resolve()
    logpath = logpath.resolve()
    kwargs["toppath"] = kwargs["toppath"].resolve()
    base_name = kwargs['outpath'].name
    os.chdir(workdir)

    for idx in range(0, nrepl):
        replica_dir = workdir / Path(str(idx))
        os.mkdir(replica_dir)
        os.chdir(replica_dir)
        kwargs['outpath'] = replica_dir / Path(base_name)
        gen_coords(**kwargs)

        for jdx, mdppath in enumerate(mdps_resolved):
            mdpname = str(mdppath.stem)
            deffnm = str(jdx) + "_" + mdpname
            # Get previous coordinates
            if jdx == 0:
                startpath = kwargs["outpath"]
            else:
                startpath = replica_dir / Path(prev_deffnm + ".gro")

            # Running GROMACS protocol
            LOGGER.info(f"running simulation {mdpname}")
            sim_runner = GMXRunner(startpath,
                                   kwargs["toppath"],
                                   mdppath,
                                   deffnm=deffnm)
            status = sim_runner.run()
            prev_deffnm = deffnm

            with open(logpath, "a") as logfile:
                if status != 0:
                    logfile.write(f"replica {idx} step {jdx} has failed with unkown gmx issue\n")
                if not sim_runner.success:
                    logfile.write(f"replica {idx} step {jdx} has failed with infinite forces\n")

                logfile.write(f"replica {idx} step {jdx} successfully completed\n")

        os.chdir(workdir)
