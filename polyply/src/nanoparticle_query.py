import argparse
import subprocess
import logging
from nanoparticle_lattice import NanoparticleModels, return_cg_nps_type
from vermouth.gmx import gro

PYTHON_II_PATH = "/home/sang/Python-2.7.9"
INSANE_SCRIPT_PATH = (
    "~/Desktop/Papers_NP/Personal_papers/polyply_paper/M3-Sterol-Parameters"
)


def generate_bilayer_process_insane(np_string: str = None) -> None:
    """ """
    command = f"{PYTHON_II_PATH}/python {INSANE_SCRIPT_PATH}/insane.py -l DPPC:4 -l DIPC:3 -l CHOL:3 -salt 0.15 -x 15 -y 10 -z 9 -d 0 -pbc cubic -sol W -o dppc-dipc-chol-insane.gro >> output.txt"

    if np_string:
        command = f"{PYTHON_II_PATH}/python {INSANE_SCRIPT_PATH}/insane.py -l DPPC:4 -l DIPC:3 -l CHOL:3 -salt 0.15 -x 15 -y 10 -z 9 -d 0 -pbc cubic -sol W -f {np_string} -o dppc-dipc-chol-insane.gro >> output.txt"

    result = subprocess.run(
        command,
        shell=True,
        universal_newlines=True,
    )
    if result.returncode == 0:
        print("Python 2 script ran successfully.")
        print(f"{result.stdout}")
    else:
        print("Python 2 script encountered an error.")


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
    # main()
    generate_bilayer_process_insane(None)
