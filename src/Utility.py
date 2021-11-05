# -*- coding: utf-8 -*-
__author__ = "He Sun"
__email__ = "wushanyun64@gmail.com"
import os
from pymatgen.io.cif import CifWriter
from src.local_features import NMR_local
import pandas as pd
from tqdm import tqdm

def struc_to_cif(structure, filename):
    """
    A function take in MP structure and output .cif files
    ----------------
    Parameter:
    structure: MP.structure
        A MP structure object with all the structural information of the crystal
    filename: str
        A str show the name of the file, exp 'Al2O3.cif'
    """
    file_dir = os.getcwd() + "/" + filename
    print("Save the file to", file_dir)
    cifwriter = CifWriter(structure)
    cifwriter.write_file(file_dir)

def features_gen(struc_tensor):
    """
    Combine the NMR values and structural parameters into one table.
    """
    table = pd.DataFrame()
    n = 0
    error_list = []
    error_message = []
    for sample in tqdm(struc_tensor):
        try:
            NMR_struc = NMR_local(sample["structure"])
            first_bond_length = pd.DataFrame.from_dict(
                NMR_struc.get_first_bond_length(), orient="index"
            )
            first_bond_angle = pd.DataFrame.from_dict(
                NMR_struc.get_first_bond_angle(), orient="index"
            )
            # l_strain = pd.DataFrame.from_dict(
            #     NMR_struc.get_longitudinal_strain(),
            #     orient="index",
            #     columns=["longitudinal_strain"],
            # )
            # s_strain = pd.DataFrame.from_dict(
            #     NMR_struc.get_shear_strain(), orient="index", columns=["shear_strain"]
            # )
            di = pd.DataFrame.from_dict(
                NMR_struc.get_DI(), orient="index", columns=["DI"]
            )
            alchemical_features = pd.DataFrame.from_dict(
                NMR_struc.get_species_features(),orient="index"
            )
            nmr = pd.DataFrame(sample["tensors"]).set_index("site_index")
            nmr = nmr.loc[:, ["max_ce","structure_index","diso","etaQ","CQ"]]
            nmr["CQ"] = abs(nmr["CQ"])  # Get absolute values for all the CQ
            sample_table = pd.concat(
                [
                    nmr,
                    first_bond_length["fbl_average"],
                    first_bond_length["fbl_std"],
                    first_bond_length['fbl_max'],
                    first_bond_length['fbl_min'],
                    first_bond_angle["fba_average"],
                    first_bond_angle["fba_std"],
                    first_bond_angle["fba_max"],
                    first_bond_angle["fba_min"],
                    # l_strain["longitudinal_strain"],
                    # s_strain["shear_strain"],
                    di["DI"],
                    alchemical_features,
                ],
                axis=1,
            )

            if table.empty:
                table = sample_table
            else:
                table = table.append(sample_table)
        except Exception as e:
            error_list.append(n)
            error_message.append(e)
        n += 1
    print(
        f"There are {len(error_list)} structures returns error. Their index are {error_list}"
    )
    print("error_messages:\n", error_message)
    return table

def get_composition(structure):
    """
    Get the atomic composition of a cetain structure
    """
    atom_list = []
    for site in structure.sites:
        atom_list.append(site.specie.symbol)
    return (list(set(atom_list)))