import numpy as np


def a():
    return


# main code executable
if __name__ == "__main__":
    # The gold nanoparticle - generate the core of the opls force field work
    AUNP_model = NanoparticleModels(
        "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/AMBER_AU/au144.gro",
        return_amber_nps_type("au144_OPLS_bonded"),
        "NP2",
        "AU",
        "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/AMBER_AU/ligand",
        ["UNK_DA2640/UNK_DA2640.itp", "UNK_12B037/UNK_12B037.itp"],
        # ["UNK_DA2640/UNK_DA2640.itp"],
        [50, 50],
        "Striped",
        ["S07", "S00"],
        ["C08", "C07"],
        3,
        "test",
        original_coordinates={
            "DA": gro.read_gro(
                "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/AMBER_AU/ligand/UNK_DA2640/UNK_DA2640.gro"
            ),
            "12B": gro.read_gro(
                "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/AMBER_AU/ligand/UNK_12B037/UNK_12B037.gro"
            ),
        },
        identify_surface=False,
    )

    AUNP_model.core_generate_coordinates()
    AUNP_model._identify_indices_for_core_attachment()
    AUNP_model._ligand_generate_coordinates()
    AUNP_model._add_block_indices()  # Not sure whether we need this now ...
    AUNP_model._generate_ligand_np_interactions()
    AUNP_model._generate_bonds()
    AUNP_model._initiate_nanoparticle_coordinates()  # doesn't quite work yet.
    # Generating output files
    AUNP_model.create_gro("gold.gro")
    AUNP_model.write_itp("gold.itp")

    # PCBM nanoparticle (Coarse-grained) - constructing the PCBM
    PCBM_ligand_gro = "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/PCBM_CG/PCBM_ligand.gro"
    ## Creating the PCBM model
    PCBM_model = NanoparticleModels(
        "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/PCBM_CG/F16.gro",
        return_cg_nps_type("F16"),
        "F16",
        "CNP",
        "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/PCBM_CG/",
        ["PCBM_ligand.itp"],
        [10],
        "Striped",
        ["C4"],
        ["N1"],
        1,
        ff_name="test",
        original_coordinates={
            "PCBM": gro.read_gro(PCBM_ligand_gro),
        },
        identify_surface=False,
    )
    # Generate PCBM
    PCBM_model.core_generate_coordinates()
    PCBM_model._identify_indices_for_core_attachment()
    PCBM_model._ligand_generate_coordinates()
    PCBM_model._add_block_indices()  # Not sure whether we need this now ...
    PCBM_model._generate_ligand_np_interactions()
    PCBM_model._generate_bonds()
    PCBM_model._initiate_nanoparticle_coordinates()  # doesn't quite work yet.
    # Generating output files
    PCBM_model.create_gro("PCBM.gro")
    PCBM_model.write_itp("PCBM.itp")
