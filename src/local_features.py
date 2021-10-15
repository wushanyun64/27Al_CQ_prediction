# -*- coding: utf-8 -*-
__author__ = "He Sun"
__email__ = "wushanyun64@gmail.com"

from pymatgen.core.structure import Structure as st
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.util.coord import get_angle
from pymatgen.analysis.bond_valence import BVAnalyzer
import numpy as np
import itertools
import statistics
from scipy.spatial import ConvexHull
import warnings
from pprint import pprint
import re
from scipy.spatial import distance_matrix

property_symbol_map={ # map the name of properties to their symbols.
    "Z":"Z",
    "mendeleev_no":"N", #Mendeleev number from definition given by Pettifor, D. G. (1984). 
    "atomic_mass":"M_a",
    "electron_affinity":"EA",
    "X":"X",
    "electronic_structure":"N_v", 
    "atomic_radius_calculated":"r_cal",
    "van_der_waals_radius":"r_vdw",
    "thermal_conductivity":"lambda",
    "melting_point":"TM",
    "boiling_point":"TB",
    "ionization_energy":"IE1", #First ionization energy
}


class NMR_local:
    """
    A class representing NMR relavent local structure parameters
    such as first order bondlength etc.

    Parameters
    -------------------------
    structure: pymatgen.structure
        The pymatgen structure obj that represent the crystal structure.
    atom: str
        Determine the atom of interest, in this project we care about "Al"
    include_oxidation:boo
        if True, decorate the structure with oxydation charge using BVAnalyzer or "by_guess".
        Very time consuming, but can increase the reliability of neighbours assignment by Voronoi tessellation.
    oxi_method: str
        When include_oxidation==True, choose which method to use. ['BV','guess']
    """

    def __init__(
        self, structure, atom="Al", include_oxidation=False, oxi_method="guess"
    ):
        self._atom = atom
        self._atom_list = self._get_atom_list(structure)

        if include_oxidation:
            if oxi_method == "BV":
                bv = BVAnalyzer()
                try:
                    oxy_structure = bv.get_oxi_state_decorated_structure(structure)
                except Exception:
                    warnings.warn(
                        """
                        Oxidation state can only be assigned for ordered structures.
                        Oxidation state can not be assigned.
                        """
                    )
                    oxy_structure = structure
            elif oxi_method == "guess":
                try:
                    oxy_structure = structure.add_oxidation_state_by_guess()
                except Exception:
                    warnings.warn(
                        """
                        Oxidation state can not be assigned.
                        """
                    )
                    oxy_structure = structure
            else:
                raise ValueError('Value for oxi_method should be "BV" or "guess".')
            self.structure = oxy_structure
            self.first_neighbours = self.get_first_coord()
        else:
            self.structure = structure
            self.first_neighbours = self.get_first_coord()

    @classmethod
    def from_cif(cls, cif, atom="Al", include_oxidation=True):
        """
        Load a cif file into NMR_local.

        Parameter
        -------------------------------
        cif: str
            The address of the cif file for the structure of interest
        include_oxidation:boo
            if True, decorate the structure with oxydation charge using BVAnalyzer
        """
        structure = st.from_file(cif)
        return cls(structure, atom, include_oxidation)

    def print_sites(self):
        """
        print the list of sites in structure

        """
        index = 0
        for i in self.structure.sites:
            pprint((index, i))
            index += 1

    def _get_atom_list(self, structure):
        """
        Get the site index list of the atom of interest ('Al' in our case)

        Parameter
        -------------------
        structure:pymatgen.structure
            The no oxydation state structure used to determine the atom list

        Return
        -------------------
        index_list: list
            A list of the index number for all the atom of interest in the structure.

        """
        index = 0
        index_list = []
        for site in structure:
            if site.specie.symbol == self._atom:
                index_list.append(index)
            index += 1
        return index_list

    def get_first_coord(self):
        """
        Create a 2D table of the first coord atoms for each atom of interest.

        Return
        --------------------
        first_coord_dict: Dataframe
            column name is the index of the atom
        """
        first_coord_dict = {}
        for index in self._atom_list:
            crystalnn = CrystalNN()
            nn_info = crystalnn.get_nn_info(self.structure, index)
            first_coord_dict[index] = nn_info
        return first_coord_dict

    def get_first_bond_length(self):
        """
        create a 2D table of the first order bond length for each atom of interest.

        Return
        -----------------
        first_bond_length_dict: dict
            A dict of first order bond length and relavent statistics (average and std) of the atoms of interest.
        """
        first_coord_dict = self.first_neighbours
        first_bond_length_dict = {}
        for center_index, first_neighbours in first_coord_dict.items():
            distances = []
            for site in first_neighbours:
                vec = site["site"].coords - self.structure[int(center_index)].coords
                distances.append(np.linalg.norm(vec))
            distances_ave = statistics.mean(distances)
            distances_std = statistics.stdev(distances)
            distances_max = max(distances)
            distances_min = min(distances)

            first_bond_length_dict[center_index] = {
                "fbl_values": distances,
                "fbl_average": distances_ave,
                "fbl_std": distances_std,
                "fbl_max":distances_max,
                "fbl_min":distances_min,
                "length": len(distances),
            }
        return first_bond_length_dict

    def get_longitudinal_strain(self):
        """
        Calculate the longitudinal strain of the chosen site. Refer to the following paper for the definition.
        J. Phys. Chem. B 2002, 106, 51, 13176–13185
        As for now, we only consider 4 5 and 6 coordinated site.
        The definition of ideal ployhedraon for different coordination numbers could be find as follows:
        Acta Cryst. (1998). B54, 766-773

        Parameters
        -----------------

        Return
        -----------------
        l_strains_dict: dict
            A dict of the longitudianl strain of the atoms of interest.
        """
        first_coord_dict = self.first_neighbours
        l_strains_dict = {}
        for center_index, first_neighbours in first_coord_dict.items():
            coord_numbers = len(first_neighbours)  # Get the number of coordination

            site_coords = []  # Here we calculate the volume of the Al polyhedron.
            for site in first_neighbours:
                site_coords.append(site["site"].coords)
            volume = ConvexHull(np.asarray(site_coords)).volume

            if coord_numbers == 4:
                r = (1.948557158 * volume) ** (
                    1.0 / 3
                )  # Calculate the bond length for a perfect tetrahedron with the same volume.
                ln_list = []  # store ln(abs(bl/r)) values
                for site in first_neighbours:
                    vec = site["site"].coords - self.structure[int(center_index)].coords
                    bl = np.linalg.norm(vec)  # calculate bond length
                    ln_list.append(abs(bl / r))
                l_strains_dict[center_index] = np.log(ln_list).sum()
            elif coord_numbers == 5:
                # Calculate the bond length for an ideal trigonal bipyramid tetrahedron 
                # with the same volume.
                r = (2.309401077 * volume) ** (1.0 / 3)
                ln_list = []
                for site in first_neighbours:
                    vec = site["site"].coords - self.structure[int(center_index)].coords
                    bl = np.linalg.norm(vec)  # calculate bond length
                    ln_list.append(abs(bl / r))
                l_strains_dict[center_index] = np.log(ln_list).sum()
            elif coord_numbers == 6:
                r = (0.75 * volume) ** (1.0 / 3)
                ln_list = []  # store ln(abs(bl/r)) values
                for site in first_neighbours:
                    vec = site["site"].coords - self.structure[int(center_index)].coords
                    bl = np.linalg.norm(vec)  # calculate bond length
                    ln_list.append(abs(bl / r))
                l_strains_dict[center_index] = np.log(ln_list).sum()
            else:
                warnings.warn(
                    "Coordination number has to be 4, 5 or 6!",
                    category=None,
                    stacklevel=1,
                    source=None,
                )
        return l_strains_dict

    def get_shear_strain(self):
        """
        Calculate the shear strain of the chosen site. Refer to the following paper for the definition.
        J. Phys. Chem. B 2002, 106, 51, 13176–13185
        As for now, we only consider 4, 5 and 6 coordinated site.
        The definition of ideal ployhedraon for different coordination numbers could be find as follows:
        Acta Cryst. (1998). B54, 766-773

        Parameters
        -----------------

        Return
        -----------------
        s_strains_dict: dict
            A dict of the longitudianl strain of the atoms of interest.
        """
        first_coord_dict = self.first_neighbours
        s_strains_dict = {}
        for center_index, first_neighbours in first_coord_dict.items():
            coord_numbers = len(first_neighbours)  # Get the number of coordination

            if coord_numbers == 4:
                angle_0 = 1.911135529  # The bond angle for perfect tetrahedron
                angles_differences = (
                    []
                )  # List of the difference between angle_0 and real bond_angles
                site_combos = itertools.combinations(first_neighbours, 2)
                for combo in site_combos:
                    v1 = (
                        combo[0]["site"].coords
                        - self.structure[int(center_index)].coords
                    )
                    v2 = (
                        combo[1]["site"].coords
                        - self.structure[int(center_index)].coords
                    )
                    angles_differences.append(
                        get_angle(v1, v2, units="radians") - angle_0
                    )
                if len(angles_differences) != 6:
                    warnings.warn(
                        f"""number of angles for tetrahedron should be 6! 
                        Current number is {len(angles_differences)}""",
                        category=None,
                        stacklevel=1,
                        source=None,
                    )
                s_strains_dict[center_index] = abs(np.tan(angles_differences)).sum()

            elif coord_numbers == 5:
                angle_0 = 1.570796327  # The first bond angle of an ideal trigonal bipyramid (90 degree).
                angle_1 = 2.094395102  # The second bond angle of an ideal trigonal bipyramid (120 degree).
                angles_differences = (
                    []
                )  # List of the difference between angle_0 and real bond_angles
                site_combos = itertools.combinations(first_neighbours, 2)
                for combo in site_combos:
                    v1 = (
                        combo[0]["site"].coords
                        - self.structure[int(center_index)].coords
                    )
                    v2 = (
                        combo[1]["site"].coords
                        - self.structure[int(center_index)].coords
                    )
                    angles_differences.append(get_angle(v1, v2, units="radians"))
                angles_differences.sort(reverse=True)
                angles_differences_0 = [
                    x - angle_0 for x in angles_differences[1:4]
                ]  # angle difference on 120 degree angles
                angles_differences_1 = [
                    x - angle_1 for x in angles_differences[4:]
                ]  # angle difference on 90 degree angles
                angles_differences = angles_differences_0 + angles_differences_1
                if len(angles_differences) != 9:
                    warnings.warn(
                        f"""number of angles for octahedron should be 9! 
                        Current number is {len(angles_differences)}""",
                        category=None,
                        stacklevel=1,
                        source=None,
                    )
                s_strains_dict[center_index] = abs(np.tan(angles_differences)).sum()

            elif coord_numbers == 6:
                angle_0 = 1.570796327  # The bond angle for perfect octahedron
                angles_differences = (
                    []
                )  # List of the difference between angle_0 and real bond_angles
                site_combos = itertools.combinations(first_neighbours, 2)
                for combo in site_combos:
                    v1 = (
                        combo[0]["site"].coords
                        - self.structure[int(center_index)].coords
                    )
                    v2 = (
                        combo[1]["site"].coords
                        - self.structure[int(center_index)].coords
                    )
                    angles_differences.append(get_angle(v1, v2, units="radians"))
                angles_differences.sort(reverse=True)
                angles_differences = [
                    x - angle_0 for x in angles_differences[3:]
                ]  # Exclude the 180 degrees.
                if len(angles_differences) != 12:
                    warnings.warn(
                        f"""number of angles for octahedron should be 12! 
                        Current number is {len(angles_differences)}""",
                        category=None,
                        stacklevel=1,
                        source=None,
                    )
                s_strains_dict[center_index] = abs(np.tan(angles_differences)).sum()
            else:
                warnings.warn(
                    "Coordination number has to be 4, 5 or 6!",
                    category=None,
                    stacklevel=1,
                    source=None,
                )
        return s_strains_dict

    def get_DI(self):
        """
        Calculate the distortion index (DI) of the chosen site. 
        Refer to the following paper for the definition.
        J. Phys. Chem. B 2002, 106, 51, 13176–13185

        Parameters
        -----------------

        Return
        -----------------
        di_dict: dict
            A dict of the DI of the atoms of interest.
        """
        first_coord_dict = self.first_neighbours
        di_dict = {}
        for center_index, first_neighbours in first_coord_dict.items():
            coord_numbers = len(first_neighbours)  # Get the number of coordination

            if coord_numbers == 4:
                angle_0 = 1.911135529  # The bond angle for perfect tetrahedron
                angles_differences = (
                    []
                )  # List of the difference between angle_0 and real bond_angles
                site_combos = itertools.combinations(first_neighbours, 2)
                for combo in site_combos:
                    v1 = (
                        combo[0]["site"].coords
                        - self.structure[int(center_index)].coords
                    )
                    v2 = (
                        combo[1]["site"].coords
                        - self.structure[int(center_index)].coords
                    )
                    angles_differences.append(
                        get_angle(v1, v2, units="radians") - angle_0
                    )
                if len(angles_differences) != 6:
                    warnings.warn(
                        f"""number of angles for octahedron should be 6! 
                        Current number is {len(angles_differences)}""",
                        category=None,
                        stacklevel=1,
                        source=None,
                    )
                di_dict[center_index] = abs(np.array(angles_differences)).sum() / (
                    angle_0 * 6
                )
            elif coord_numbers == 5:
                angle_180 = 3.141592653
                angle_90 = angle_180 / 2
                angle_120 = angle_180 * 2 / 3
                angles_differences = []
                site_combos = itertools.combinations(first_neighbours, 2)
                for combo in site_combos:
                    v1 = (
                        combo[0]["site"].coords
                        - self.structure[int(center_index)].coords
                    )
                    v2 = (
                        combo[1]["site"].coords
                        - self.structure[int(center_index)].coords
                    )
                    angles_differences.append(get_angle(v1, v2, units="radians"))
                angles_differences.sort(reverse=True)
                angle_diff_180 = [
                    x - angle_180 for x in angles_differences[0:1]
                ]  # Angle difference for 180 degrees angle
                angle_diff_120 = [
                    x - angle_120 for x in angles_differences[1:4]
                ]  # Angle difference for 120 degrees angle
                angle_diff_90 = [
                    x - angle_90 for x in angles_differences[4:]
                ]  # Angle difference for 90 degrees angle
                angles_differences = angle_diff_180 + angle_diff_120 + angle_diff_90
                if len(angles_differences) != 10:
                    warnings.warn(
                        f"""number of angles for octahedron should be 10! 
                        Current number is {len(angles_differences)}""",
                        category=None,
                        stacklevel=1,
                        source=None,
                    )
                di_dict[center_index] = (
                    abs(np.array(angles_differences)).sum() / 18.8495559
                )
            elif coord_numbers == 6:
                angle_0 = 1.570796327  # The bond angle for perfect tetrahedron
                angles_differences = (
                    []
                )  # List of the difference between angle_0 and real bond_angles
                site_combos = itertools.combinations(first_neighbours, 2)
                for combo in site_combos:
                    v1 = (
                        combo[0]["site"].coords
                        - self.structure[int(center_index)].coords
                    )
                    v2 = (
                        combo[1]["site"].coords
                        - self.structure[int(center_index)].coords
                    )
                    angles_differences.append(get_angle(v1, v2, units="radians"))
                angles_differences.sort(reverse=True)
                angles_differences = [
                    x - angle_0 for x in angles_differences[3:]
                ]  # Exclude the 180 degrees.
                if len(angles_differences) != 12:
                    warnings.warn(
                        f"number of angles for octahedron should be 12! Current number is {len(angles_differences)}",
                        category=None,
                        stacklevel=1,
                        source=None,
                    )
                di_dict[center_index] = abs(np.array(angles_differences)).sum() / (
                    angle_0 * 12
                )
            else:
                warnings.warn(
                    "Coordination number has to be 4, 5 or 6!",
                    category=None,
                    stacklevel=1,
                    source=None,
                )
        return di_dict

    def get_first_bond_angle(self):
        """
        create a 2D table of the first order bond angles for each atom of interest.

        Return
        -----------------
        first_bond_angle_dict: dict
            A dict of first order bond angle of the atom of interest.
        """
        first_coord_dict = self.first_neighbours
        first_bond_angle_dict = {}
        for center_index, first_neighbours in first_coord_dict.items():
            angles = []
            site_combos = itertools.combinations(first_neighbours, 2)
            for combo in site_combos:
                v1 = combo[0]["site"].coords - self.structure[int(center_index)].coords
                v2 = combo[1]["site"].coords - self.structure[int(center_index)].coords
                angles.append(get_angle(v1, v2, units="degrees"))
            angles.sort(reverse=True)
            angles_ave = statistics.mean(angles)
            angles_std = statistics.stdev(angles)
            angles_max = max(angles)
            angles_min = min(angles)
            first_bond_angle_dict[center_index] = {
                "fba_values": angles,
                "fba_average": angles_ave,
                "fba_std": angles_std,
                "fba_max":angles_max,
                "fba_min":angles_min,
                "length": len(angles),
            }
        return first_bond_angle_dict

    def get_second_coord(self):
        """
        Create a 2D table of the second coord atoms for each atom of interest.

        Return
        --------------------
        first_coord_dict: Dataframe
            column name is the index of the atom
        """
        second_coord_dict = {}
        for index in self._atom_list:
            crystalnn = CrystalNN()
            all_nn = crystalnn.get_all_nn_info(self.structure)

            nearneighbors = NearNeighbors()
            second_coord_info = nearneighbors._get_nn_shell_info(
                self.structure, all_nn, index, 2
            )
            second_coord_info = sorted(second_coord_info, key=lambda i: i["site_index"])
            second_coord_dict[index] = second_coord_info
        return second_coord_dict


    def get_species_features(self):
        """[summary]
        """
        first_coord_dict = self.first_neighbours
        features_dict = {}
        for center_index, first_neighbours in first_coord_dict.items():

            # polyhedron properties including all the properties of all sites (center included) 
            # in the polyhedron. 
            N = len(first_neighbours)
            ploy_properties = [] 
            neighbour_coords = []
            for neighbour in first_neighbours:
                neighbour_properties = self._get_site_properties(neighbour['site'])
                ploy_properties.append(neighbour_properties)
                neighbour_coords.append(neighbour['site'].coords)
            neighbour_coords = np.array(neighbour_coords)
            ploy_properties = np.array(ploy_properties)
            species_features = self._species_features_calculator(ploy_properties, center_index,neighbour_coords,N)
            features_dict[center_index]=species_features
        return(features_dict)
    
    @staticmethod
    def _get_site_properties(site):
        """
        something here.

        Note:electronic_structure here will be processed into representing the number of valence 
        electrons (N_v).

        Args:
            site ([type]): [description]
        """
        property_list = []
        for k in property_symbol_map:
            if k == "electronic_structure": # Use e struc to get number of valence electrons.
                e_struc = getattr(site.species.elements[0],k)
                prog = re.compile(r'\.[0-9][s,p,d,f,g,h,i]([0-9])')
                N_v = sum(map(int,prog.findall(e_struc)))
                property_list.append(N_v)
            else:
                property_list.append(getattr(site.species.elements[0],k))
        return(property_list)

    def _species_features_calculator(self,properties_matrix, center_index,neighbour_coords,N):
        """[summary]

        Args:
            properties_matrix ([type]): [description]
            structure ([type]): [description]
            center_index ([type]): [description]
        """
        #get statistics of the properties values
        means = properties_matrix.mean(axis=0)
        std = properties_matrix.std(axis=0)
        max_ = properties_matrix.max(axis=0)
        min_ = properties_matrix.min(axis=0)
        
        #get average deviation of the properties normalized by r_cn 
        # (distance between center and neighbour)
        center_site = self.structure[center_index]
        center_coords = self.structure[center_index].coords
        center_properties = self._get_site_properties(center_site)
        r_cn_rev_list = np.reciprocal(np.linalg.norm(neighbour_coords - center_coords,axis=1))
        dev_matrix = (properties_matrix-center_properties)*(1/N)
        properties_dev_r = (dev_matrix.T*r_cn_rev_list).T.sum(axis=0)

        #get alchemical matrix
        #matrix of all properties including center atom
        properties_all = np.vstack([properties_matrix,center_properties])  
        columns = properties_all.shape[1]
        matrix_list = []
        for i in range(columns):
            matrix_list.append(np.outer(properties_all[:, i],properties_all[:, i]))
        coords_all = np.vstack([neighbour_coords,center_coords])
        dis_matrix_rev = np.reciprocal(distance_matrix(coords_all,coords_all))
        #Fill diagonal of reciprocal distance matrix with P_i^2 
        np.fill_diagonal(dis_matrix_rev,1) 
        #get the singular values from alchemical matrix 
        alchemical_s = []
        for pm in matrix_list: #pm property matrix
            alchemical_matrix = np.multiply(pm,dis_matrix_rev)
            svd = np.linalg.svd(alchemical_matrix,full_matrices=False)
            #store the first 5 singular values
            alchemical_s.append(svd[1][:5])
        
        # Get a flatened vector of all property features
        statistics_flat = np.hstack([means,std,max_,min_])
        alchemical_s_flat = np.array(alchemical_s).flatten()
        properties_dev_flat = np.hstack([properties_dev_r])
        result = np.hstack([statistics_flat,properties_dev_flat,alchemical_s_flat])
        return(result)