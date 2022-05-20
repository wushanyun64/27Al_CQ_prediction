# -*- coding: utf-8 -*-
__author__ = "He Sun"
__email__ = "wushanyun64@gmail.com"
import os
from pymatgen.io.cif import CifWriter
from src.local_features import NMR_local
import pandas as pd
from tqdm import tqdm
import numpy as np

from collections import defaultdict
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import math

import seaborn as sns
import matplotlib.pyplot as plt

scores_list = [
    "cv_r2_mean",
    "cv_RMSE_mean",
    "cv_MAE_mean",
    "train_r2",
    "train_RMSE",
    "train_MAE",
    "test_r2",
    "test_RMSE",
    "test_MAE",
]


def learning_curve_hyperparam(
    model, X_train, y_train, X_test, y_test, param_name, param_values
):
    """
    Plot a learning curve based on certain hyperparam for the model (random forest in this case),
    go through all the values in the param_values list and compute the train, test and cv scores.
    -----------
    Parameters
    param_name: the name of the feature to get the learning curve on.
    param_values: A series of values for the faeture.
    """
    result_dict = defaultdict(list)

    for v in tqdm(param_values):
        setattr(model, param_name, v)
        scores = fit_model(model, X_train, y_train, X_test, y_test)

        result_dict["v"].append(v)
        for score_name in scores_list:
            result_dict[score_name].append(scores[score_name])
    return result_dict


def learning_curve_samplesize(model, small_sets, feature_names):
    """
    Plot a learning curve based on difference sample sizes,
    go through all the small sample sets in small_sets and compute the train, test and cv scores.
    -----------
    Parameters
    small_sets: list of smaller sample randomly selected from the whole dataset.
    feature_names:list of features to consider in the model.
    """
    result_dict = defaultdict(list)

    for set_ in tqdm(small_sets):
        # split y and x
        y = set_["CQ"]
        x = set_[feature_names]

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=20
        )

        scores = fit_model(model, X_train, y_train, X_test, y_test)

        result_dict["sample_size"].append(x.shape[0])
        for score_name in scores_list:
            result_dict[score_name].append(scores[score_name])
    return result_dict


def fit_model(model, X_train, y_train, X_test, y_test):
    """
    fit a model to the given X and y, return the train, test and cv scores in a dictionary.
    """
    scores = {}
    model.fit(X_train, y_train)

    # Get train scores
    y_train_predict = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_predict)
    train_RMSE = math.sqrt(mean_squared_error(y_train, y_train_predict))
    train_MAE = mean_absolute_error(y_train, y_train_predict)

    # Get cv scores
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=["neg_mean_absolute_error", "neg_mean_squared_error", "r2"],
    )
    cv_r2_mean = cv_scores["test_r2"].mean()
    cv_RMSE_mean = np.sqrt(-cv_scores["test_neg_mean_squared_error"]).mean()
    cv_MAE_mean = -cv_scores["test_neg_mean_absolute_error"].mean()

    # Get test scores
    y_rf = model.predict(X_test)
    test_r2 = r2_score(y_test, y_rf)
    test_RMSE = math.sqrt(mean_squared_error(y_test, y_rf))
    test_MAE = mean_absolute_error(y_test, y_rf)

    scores["cv_r2_mean"] = cv_r2_mean
    scores["cv_RMSE_mean"] = cv_RMSE_mean
    scores["cv_MAE_mean"] = cv_MAE_mean
    scores["train_r2"] = train_r2
    scores["train_RMSE"] = train_RMSE
    scores["train_MAE"] = train_MAE
    scores["test_r2"] = test_r2
    scores["test_RMSE"] = test_RMSE
    scores["test_MAE"] = test_MAE
    return scores


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
            first_compo = pd.DataFrame.from_dict(
                NMR_struc.get_first_coord_compo(), orient="index"
            )
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
                NMR_struc.get_species_features(), orient="index"
            )
            nmr = pd.DataFrame(sample["tensors"]).set_index("site_index")
            nmr = nmr.loc[:, ["max_ce", "structure_index", "diso", "etaQ", "CQ"]]
            nmr["CQ"] = abs(nmr["CQ"])  # Get absolute values for all the CQ
            sample_table = pd.concat(
                [
                    first_compo,
                    nmr,
                    first_bond_length["fbl_average"],
                    first_bond_length["fbl_std"],
                    first_bond_length["fbl_max"],
                    first_bond_length["fbl_min"],
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
    # get rid of nmr tensors for symmetrically equal sites.
    table = table[pd.notna(table["fbl_average"])]
    return table


def get_composition(structure):
    """
    Get the atomic composition of a cetain structure
    """
    atom_list = []
    for site in structure.sites:
        atom_list.append(site.specie.symbol)
    return list(set(atom_list))


def reg_plot(y, yhat, y_name, yhat_name):
    result = {}
    result["y"] = y
    result["yhat"] = yhat
    result = pd.DataFrame(result)

    result = result.rename(columns={"y": y_name, "yhat": yhat_name})

    print(result.columns)

    # plot the correlation
    sns.set_style("ticks")
    plt.figure(figsize=(10, 8))
    plt.rcParams["font.size"] = "20"
    sns.regplot(
        x=y_name,
        y=yhat_name,
        data=result,
        ci=None,
        scatter_kws={"color": "black"},
        line_kws={"color": "red"},
    )
    sns.despine()
    plt.show()
