import pandas as pd
from src.Utility import get_composition
from dscribe.descriptors import SOAP
from itertools import combinations_with_replacement
from scipy.sparse import vstack
from random import sample
from pymatgen.io import ase
from tqdm import tqdm


def get_species(data):
    """
    List all the atomic species in the data set.

    Args:
        data (dict): structure_tensors dict.

    Returns:
        species_list: list od all occurred species.
        combinations: list of compositions for each site. 
    """
    species_list = []
    combinations = []
    for structure_tensor in data:
        compo = get_composition(structure_tensor['structure'])
        species_list.extend(compo)
        combinations.append(compo)
    species_list = list(set(species_list))
    return (species_list,combinations)

def getSOAPforStruct(structure_tensor, species, rCut=3, nMax=12, Lmax=9):
    """
    Get the SOAP descriptor for a given structure.

    Args:
        structure_tensor (dict): dict of structure and tensor information.
        species (list): list of species in the data set that current structure belongs to.
        rCut (int, optional): A cutoff for local region in angstroms. 
                              Should be bigger than 1 angstrom for the gto-basis. Defaults to 3.
        nMax (int, optional): The number of radial basis functions. Defaults to 12.
        Lmax (int, optional): The maximum degree of spherical harmonics. Defaults to 9.

    Returns:
        [type]: [description]
    """

    # Get structure and atom coord
    atoms = ase.AseAtomsAdaptor.get_atoms(structure_tensor["structure"])

    # initialize SOAP
    species = species
    rcut = rCut
    nmax = nMax
    lmax = Lmax

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=True,
        rcut=rcut,
        nmax=nmax,
        lmax=lmax,
        crossover=True,
        sparse=True,
    )

    # Get the coordination of Al atom to create SOAP on. 
    coord = []
    for tensor in structure_tensor["tensors"]:
        coord.append(tensor["site_coord"])

    location_dict = {}
    for combo in combinations_with_replacement(species, 2):
        loc = soap.get_location(combo)
        location_dict[combo] = loc

    x = soap.create(atoms, positions=coord)

    return (x.tocsr(), location_dict)

def TensorSOAPcombo(structure_tensor, nmr_key, species):
    """[summary]

    Args:
        structure_tensor ([type]): [description]
        nmr_key ([type]): [description]
        species ([type]): [description]

    Returns:
        [type]: [description]
    """
    x, loc = getSOAPforStruct(structure_tensor, species)
    y = []
    for tensor in structure_tensor["tensors"]:
        y.append([tensor[nmr_key], structure_tensor["structure"], tensor["site_index"]])
    return (x, y, loc)

# getXY when choose sparse matrix in soap
def getXY(structures_tensors,species):
    #loctaion info for slice the soap vector, None when choose other descriptor
    loc = None
    X=None
    y=[]
    
    for item in tqdm(structures_tensors):
        soap, nmr_param, loc = TensorSOAPcombo(item,'CQ',species = species)
        if X==None:
            X =soap
        else:
            X = vstack([X,soap])
        y.extend(nmr_param)
    
    y = pd.DataFrame(y,columns=['nmr','structure','site_index'])
    
    return(X,y,loc)