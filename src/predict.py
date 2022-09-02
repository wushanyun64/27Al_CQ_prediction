# -*- coding: utf-8 -*-
import joblib

from src.data.structure_tensors_gen import get_structure_tensor
from src.data.structure_tensors_modifier import *
from src.features_gen import concat_features_and_nmr
from src.features_gen import table_clean


class predict:
    def __init__(self, model_name, feature_type, structure_list):
        """
        Utility obj to construct the pipeline to make predictions with pretrained models.

        Parameters
        ---------------------------
        model_name: str
            name of the trained model that can be used, model can be find in ./models.
        feature_type: str
            A string that indicates which feature set to use, either "struc" or "struc+ele".
            The type of feature should match the type of trained model.
        structure_list: list
            A list of structures for prediction. The structures stored need to be compatible with
            pymatgen.Structure, otherwise an error will raise.
        """
        self.feat_type = feature_type
        # import trained models
        self.model = joblib.load("../models/" + model_name)

        print("Loading structures")
        self.structure_tensors = []
        for compound in tqdm(structure_list):
            structure = compound["structure"]
            efg = compound["efg"]
            cs = compound["cs"]
            structure_tensor = get_structure_tensor(structure, efg, cs)
            self.structure_tensors.append(structure_tensor)

        print("Preprocessing...")
        self.prep()

    def model_predict(self):
        """
        The function to make the prediction and return the original labels (y) and
        predicted labels (y_pre).
        """
        features = self.feature_generation()
        if self.feat_type == "struc":
            x = features.loc[:, "fbl_average":"DI"]

        elif self.feat_type == "struc+ele":
            x = features.loc[:, "fbl_average":]
        else:
            raise ValueError(
                """The feature_type should be either 'struc' or 'struc+ele'."""
            )
        y = features["CQ"].to_list()
        y_pre = self.model.predict(x)
        return y, y_pre

    def feature_generation(self):
        """
        Helper function to call feature generation functions.
        """
        features = concat_features_and_nmr(self.structure_tensors)
        features.reset_index(drop=True, inplace=True)
        features = table_clean(features)
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
        self.structure_tensors = add_oxi_state_by_guess(self.structure_tensors)

    def _filter_by_coordination(self):
        self.structure_tensors = get_n_coord_tensors(
            self.structure_tensors, coord=[4, 5, 6]
        )

    def _append_coordination(self):
        self.structure_tensors = append_coord_num(self.structure_tensors)

    def _append_chemical_env(self):
        self.structure_tensors = append_ce(self.structure_tensors)

    def _filter_by_chemical_env(self):
        """Should be used after self._append_chemical_env()"""
        chemenv_filter = filter_ce(self.structure_tensors)
        self.structure_tensors = chemenv_filter["filtered"]
