# -*- coding: utf-8 -*-
__author__ = "He Sun"
__email__ = "wushanyun64@gmail.com"

from tqdm import tqdm
import copy
import numpy as np
from src.local_features import NMR_local
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from matminer.featurizers.site.fingerprint import ChemEnvSiteFingerprint
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    MultiWeightsChemenvStrategy,
)


def apply_filters(structure_tensors, filters):
    """
    Apply the defined filters on structur_tensors obj in sequence.

    Args:
        structure_tensors (list): list of structure_tensors obj to be filterd.
        filters (list): list of fliter functions to be applied.
    """
    structure_tensors_filtered = copy.deepcopy(structure_tensors)
    for filter in filters:
        structure_tensors_filtered = filter(structure_tensors_filtered)
    return structure_tensors_filtered

def add_oxi_state_by_guess(structure_tensors):
    """
    Get the oxidation state for each structure in structure_tensors obj with add_oxidation_state_by_guess()
    Args:
        structure_tensors (list): list of structure_tensors obj.
    """
    print("Add oxidation state by guess.")
    for sample in tqdm(structure_tensors):
        sample['structure'].add_oxidation_state_by_guess()
    return(structure_tensors)

def get_n_coord_tensors(structure_tensors, coord=[4, 5, 6]):
    """
    Get the structure tensors infomation based on Al coordination numbers.

    Args
    --------------------------
    structures_tensors: list
        The list of structures and tensors data.
    coord: list
        The list of coordinaton number to get.
    **kwargs: Additional keywords passed to NMR_local
    Return
    -------------------------
    n_coord_filtered: list
        The list of structures and tensors data with defined Al coordination numbers.
    """
    print("Filter structures based coordination number.")
    n = 0
    n_coord_filtered = []
    for sample in tqdm(structure_tensors):
        NMR_struc = NMR_local(sample["structure"])
        first_neighbours = NMR_struc.first_neighbours
        coord_num_list = [len(first_neighbours[k]) for k in first_neighbours]
        within = True
        for v in coord_num_list:
            if v not in coord:
                within = False
        if within:
            n_coord_filtered.append(sample)
        n += 1
    print(f"num of structures with {coord} coords only: {len(n_coord_filtered)}")
    return n_coord_filtered


def append_coord_num(structure_tensors):
    """
    Append the coordination number for each site in "tensors".

    Args
    --------------------------
    structures_tensors: list
        The list of structures and tensors data.
    **kwargs: 
        Additional keywords passed to NMR_local
    Return
    -------------------------
    structure_tensors: list
        The list of structures and tensors data with coord num included in "tensors".
    """
    print("Append coordination numbers to tensor list.")
    for sample in tqdm(structure_tensors):
        NMR_struc = NMR_local(sample["structure"])
        first_neighbours = NMR_struc.first_neighbours
        for tensor in sample["tensors"]:
            tensor["coord_num"] = len(first_neighbours[tensor["site_index"]])
    return structure_tensors


def get_Al_O_structs(structure_tensors):
    """
    Get all the structures whose Al is only bonded to O.

    Args:
    structures_tensors (list): list of structure_tensors obj to be filterd
    """
    crystalnn = CrystalNN()
    Al_O_struc = []
    data_copy = copy.deepcopy(structure_tensors)
    for compound in tqdm(data_copy):
        i = 0
        for tensor in compound["tensors"]:
            nn_info = crystalnn.get_nn_info(
                compound["structure"], int(tensor["site_index"])
            )
            neighbours = []
            for site_info in nn_info:
                neighbours.append(site_info["site"].specie)
            if set(neighbours) == {Element("O")}:
                i += 1
        if i == len(compound["tensors"]):
            Al_O_struc.append(compound)
    return Al_O_struc


def append_ce(structure_tensors):
    """
    [Append the ChemEnv info for each site in the 'Tensor' section of all structure_tensors.]

    Args:
        structures_tensors (list): list of structure_tensors obj.
    """
    print("Append ChemEnv info to tensor list.")
    cetypes = {
        3: ["TL:3", "TY:3", "TS:3"],
        4: ["T:4", "S:4", "SY:4", "SS:4"],
        5: ["PP:5", "S:5", "T:5"],
        6: ["O:6", "T:6", "PP:6"],
    }
    lgf = LocalGeometryFinder()
    lgf.setup_parameters(
        centering_type="centroid",
        include_central_site_in_centroid=True,
        structure_refinement=lgf.STRUCTURE_REFINEMENT_NONE,
    )

    for compound in tqdm(structure_tensors):
        structure = compound["structure"]
        for tensor in compound["tensors"]:
            center_index = tensor["site_index"]
            coord_num = tensor["coord_num"]
            envfingerprint = ChemEnvSiteFingerprint(
                cetypes[coord_num],
                MultiWeightsChemenvStrategy.stats_article_weights_parameters(),
                lgf,
            )
            labels = envfingerprint.feature_labels()
            ce_result = envfingerprint.featurize(structure, center_index)
            maximum = np.max(ce_result)
            max_index = np.argmax(ce_result)
            max_ce = labels[max_index]
            tensor["max_ce"] = max_ce
            tensor["max_ce_value"] = maximum

    return structure_tensors


def filter_ce(structure_tensors, env_symbols=["T:4", "T:5", "O:6"]):
    """
    Filter the structure_tensors list based on local environments.
    For example: env_symbols = ["T:4"] means we only consider tetrahedron sites.
    Note: if the score of selected chemenv is too low (<0.5), the structure is also
    ruled out becase of low reliability of chemenv assignment in this case.
    Args:
        structures_tensors (list): list of structure_tensors obj to be filterd
        env_symbols(list): list of wanted chemical environments.
    """
    print("Filter structure_tensors based on ChemEnv info.")
    chemenv_filtered = []
    chemenv_outliers = []
    for sample in tqdm(structure_tensors):
        env_list = [t["max_ce"] for t in sample["tensors"]]

        within = True
        for t in sample["tensors"]:
            if t["max_ce_value"] < 0.5:
                within = False
        for env in env_list:
            if env not in env_symbols:
                within = False

        if within:
            chemenv_filtered.append(sample)
        else:
            chemenv_outliers.append(sample)

    print(f"num of structures with chemenv {env_symbols} only: {len(chemenv_filtered)}")
    return {"filtered": chemenv_filtered, "outliers": chemenv_outliers}
