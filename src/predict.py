# -*- coding: utf-8 -*-
import joblib

from src.data.structure_tensors_gen import get_structure_tensor
from src.data.structure_tensors_modifier import *
from src.features_gen import concat_features_and_nmr


class predict:
    """_summary_"""

    def __init__(self, model_name, feature_type, structure_list):
        """_summary_

        Parameters
        ---------------------------
        model_name (_type_): _description_
        feature_type: str
            A string that indicates which feature set to use, either "struc" or "struc+ele".
            The type of feature should match the type of trained model.
        structure_list: list
            _description_
        """
        self.feat_type = feature_type
        # import trained models
        self.model = joblib.load("../models/" + model_name)

        print("Loading structures")
        self.structure_tensors = []
        for compound in tqdm(structure_list):
            structure = compound
            efg = None
            cs = None
            structure_tensor = get_structure_tensor(structure, efg, cs)
            self.structure_tensors.append(structure_tensor)

        print("Preprocessing...")
        self.prep()

    def model_predict(self):
        """_summary_"""
        features = self.feature_generation()
        if self.feat_type == "struc":
            x = features.loc[:, "fbl_average":"DI"]
        elif self.feat_type == "struc+ele":
            x = features.loc[:, "fbl_average":]
        else:
            raise ValueError(
                """The feature_type should be either 'struc' or 'struc+ele'."""
            )
        y = self.model.predict(x)
        return y

    def feature_generation(self):
        features = concat_features_and_nmr(self.structure_tensors)
        features.reset_index(drop=True, inplace=True)
        return features

    def prep(self):
        """
        A helper function that takes care of data prepare process.
        """
        self._add_oxy()
        self._append_coordination()
        self._filter_by_coordination()
        self._append_chemical_env()
        self._filter_by_chemical_env()

    def _add_oxy(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        self.structure_tensors = add_oxi_state_by_guess(self.structure_tensors)

    def _filter_by_coordination(self):
        """_summary_"""
        self.structure_tensors = get_n_coord_tensors(
            self.structure_tensors, coord=[4, 5, 6]
        )

    def _append_coordination(self):
        """_summary_"""
        self.structure_tensors = append_coord_num(self.structure_tensors)

    def _append_chemical_env(self):
        """_summary_"""
        self.structure_tensors = append_ce(self.structure_tensors)

    def _filter_by_chemical_env(self):
        """Should be used after self._append_chemical_env()"""
        chemenv_filter = filter_ce(self.structure_tensors)
        self.structure_tensors = chemenv_filter["filtered"]
