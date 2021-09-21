# -*- coding: utf-8 -*-
__author__ = "He Sun"
__email__ = "wushanyun64@gmail.com"

from tqdm import tqdm
from pymatgen.core.structure import Structure as ST
from pymatgen.core.sites import Site
from pymatgen.analysis.nmr import ChemicalShielding
from pymatgen.analysis.nmr import ElectricFieldGradient
import numpy as np


def get_structure_tensors(data):
    """
    Covert the original 'data' complex (see Args for details) into a dict with structure
    and NMR parameters calculated from raw tensor.

    Return:
    structure_tensor = {
        'structure': pymatgen.structure
        'tensors : {
            'diso',
            'csa',
            'csa_reduced',
            'etacs',
            'etaQ',
            'CQ',
            'site_index',
            'site_coord',
        }
    }

    Args:
        data (list): list of data containing structure and NMR raw tensor.
    """
    compounds = []
    #     c_index=0
    for compound in tqdm(data, position=0):
        if compound == {}:
            continue
        tensors = []
        site_index = 0
        structure = ST.from_dict(compound["structure"])
        #         structure.add_oxidation_state_by_guess() ##Very time comsuing, comment out for now
        for site in structure.sites:
            lengthes = []
            if site.species_string[:2] == "Al":
                # add nmr tensor informations
                cs_origin = compound["cs"][site_index]
                cs_symmetric = 0.5 * (cs_origin + np.transpose(cs_origin))
                cs = ChemicalShielding(cs_symmetric)
                efg_origin = compound["efg"][site_index]
                efg = ElectricFieldGradient(efg_origin)
                tensor = {
                    "diso": cs.haeberlen_values[0],
                    "csa": cs.haeberlen_values[1],
                    "csa_reduced": cs.haeberlen_values[2],
                    "etacs": cs.haeberlen_values[3],
                    "etaQ": efg.asymmetry,
                    "CQ": efg.coupling_constant(specie="Al"),
                    "site_index": site_index,
                    "site_coord": list(site.coords),
                }
                tensors.append(tensor)
            site_index += 1
        compounds.append({"structure": structure, "tensors": tensors})
    return compounds
