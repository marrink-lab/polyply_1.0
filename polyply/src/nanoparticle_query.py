import typing as t
import subprocess
import logging
from nanoparticle_lattice import NanoparticleModels, return_cg_nps_type
from vermouth.gmx import gro

PYTHON_II_PATH = "/home/sang/Python-2.7.9"
INSANE_SCRIPT_PATH = (
    "~/Desktop/Papers_NP/Personal_papers/polyply_paper/M3-Sterol-Parameters"
)
GROMACS_PATH = "home/sang/Desktop/gromacs-2022.5/build/bin/gmx"


def generate_bilayer_process_insane(np_string: str = t.Union[None, str]) -> None:
    """
    Use the local python2 module to generate a bilayer using insane.

    If used as a default (None), then we will just have a default mixed martini bilayer.

    If we have a valid np_string, then we will have the option to add in a nanoparticle within
    the bilayer with the required proportioned bilayer.
    """
    # You can modify these paths to suit your directory structure
    PYTHON_II_PATH = "/home/sang/Python-2.7.9"
    INSANE_SCRIPT_PATH = (
        "~/Desktop/Papers_NP/Personal_papers/polyply_paper/M3-Sterol-Parameters"
    )
    command = f"{PYTHON_II_PATH}/python {INSANE_SCRIPT_PATH}/insane.py -l DPPC:4 -l DIPC:3 -l CHOL:3 -salt 0.15 -x 15 -y 10 -z 9 -d 0 -pbc cubic -sol W -o dppc-dipc-chol-insane.gro >> output.txt"
    if np_string:
        command = f"{PYTHON_II_PATH}/python {INSANE_SCRIPT_PATH}/insane.py -l DPPC:4 -l DIPC:3 -l CHOL:3 -salt 0.15 -x 15 -y 10 -z 9 -d 0 -pbc cubic -sol W -f {np_string} -center -o dppc-dipc-chol-insane-np.gro >> output.txt"

    result = subprocess.run(
        command,
        shell=True,
        universal_newlines=True,
    )
    if result.returncode == 0:
        logging.info("Python 2 script ran successfully.")
        logging.info(f"{result.stdout}")
    else:
        logging.info("Python 2 script encountered an error.")


def remove_pbc_np(removed_pbc_np: str, np_gro: str):
    """
    Ensure that the pbc is removed from the nanoparticle before
    the gromacs nanoparticle gro
    """
    command = f"{GROMACS_PATH} trjconv -f {np_gro} -center -pbc whole -s minim.tpr -o {removed_pbc_np}"
    result = subprocess.run(
        command,
        shell=True,
        universal_newlines=True,
    )
    if result.returncode == 0:
        logging.info("Python 2 script ran successfully.")
        logging.info(f"{result.stdout}")
    else:
        logging.info("Python 2 script encountered an error.")


def generate_pcbm_nanoparticle(input_gro, output_dir):
    # Construct PCBM nanoparticle
    PCBM_model = NanoparticleModels(
        return_cg_nps_type("F16"),
        "F16",
        "CNP",
        output_dir,
        ["PCBM_ligand.itp"],
        [1],
        "Striped",
        ["C4"],
        ["N1"],
        1,
        ff_name="test",
        original_coordinates={
            "PCBM": gro.read_gro(input_gro),
        },
        identify_surface=False,
    )

    # Generate PCBM
    PCBM_model.core_generate_coordinates()
    PCBM_model._identify_indices_for_core_attachment()
    PCBM_model._ligand_generate_coordinates()
    PCBM_model._add_block_indices()
    PCBM_model._generate_ligand_np_interactions()
    PCBM_model._generate_bonds()
    PCBM_model._initiate_nanoparticle_coordinates()

    # Generating output files
    PCBM_model.create_gro(f"{output_dir}/PCBM.gro")
    PCBM_model.write_itp(f"{output_dir}/PCBM.itp")


# def main():
#    parser = argparse.ArgumentParser(description="Generate PCBM nanoparticle.")
#
#    # Define command-line arguments
#    parser.add_argument(
#        "--input-gro",
#        required=True,
#        help="Path to the input GRO file for PCBM_ligand.",
#    )
#
#    parser.add_argument(
#        "--output-dir",
#        required=True,
#        help="Path to the output directory for generated files.",
#    )
#
#    args = parser.parse_args()
#
#    # Call the function to generate the PCBM nanoparticle
#    generate_pcbm_nanoparticle(args.input_gro, args.output_dir)


if __name__ == "__main__":
    generate_bilayer_process_insane(None)
    generate_bilayer_process_insane(
        "/home/sang/Desktop/git/polyply_1.0/polyply/src/PCBM_simulation/out.gro"
    )
