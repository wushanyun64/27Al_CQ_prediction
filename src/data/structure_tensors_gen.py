# -*- coding: utf-8 -*-
__author__ = "He Sun"
__email__ = "wushanyun64@gmail.com"

from tqdm import tqdm
from pymatgen.core.structure import Structure as ST
from pymatgen.analysis.nmr import ChemicalShielding
from pymatgen.analysis.nmr import ElectricFieldGradient
import numpy as np


def get_structure_tensor(structure, efg, cs=None):
    """
    Covert the original 'data' complex (see Args for details) into a dict with structure
    and NMR parameters calculated from raw tensor.

    Parameters
    ----------------------------
    structure: dict
        Material structure data stored in a dictionary. The format needs to be
        compatible with pymatgen.structure.
    efg: list
        A list of 3*3 efg tensors for each site in the structure, the sites and tensors needs to
        follow the same order. If efg==None, the calculated efg param will be all 0.
    cs: list
        A list of 3*3 csa tensors for each site in the structure, the sites and tensors needs to
        follow the same order. If cs==None, the calculated cs param will be all 0.

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
            'structure_index',
            'site_coord',
        }
    }

    Args:
        data (list): list of data containing structure and NMR raw tensor.
    """
    # structure_tensor = []
    # structure_index = 0
    # for compound in tqdm(data, position=0):
    tensors = []
    structure = ST.from_dict(structure)

    for i, site in enumerate(structure.sites):
        if site.species_string[:2] == "Al":
            # Initialize all the nmr params as 0
            diso = csa = csa_reduced = eta_cs = etaQ = cq = 0
            if cs:
                cs_origin = cs[i]
                cs_symmetric = 0.5 * (cs_origin + np.transpose(cs_origin))
                cs_tensor = ChemicalShielding(cs_symmetric)
                diso = cs_tensor.haeberlen_values[0]
                csa = cs_tensor.haeberlen_values[1]
                csa_reduced = cs_tensor.haeberlen_values[2]
                eta_cs = cs_tensor.haeberlen_values[3]
            if efg:
                efg_origin = efg[i]
                efg_tensor = ElectricFieldGradient(efg_origin)
                etaQ = efg_tensor.asymmetry
                cq = efg_tensor.coupling_constant(specie="Al")
            tensor = {
                "diso": diso,
                "csa": csa,
                "csa_reduced": csa_reduced,
                "etacs": eta_cs,
                "etaQ": etaQ,
                "CQ": cq,
                "site_index": i,
                "site_coord": list(site.coords),
            }
            tensors.append(tensor)

    return {"structure": structure, "tensors": tensors}
