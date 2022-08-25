# -*- coding: utf-8 -*-
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.Utility import reg_plot


def model_train(X, y, model, param, n_iter=10, cv=10):
    """
    A helper function that select the model's hyperparameters using
    RandomizedSearchCV from scikit-learn.

    Parameters
    ------------------------
    X: array like
        Training data.
    y: array like
        Training labels. y==CQ in our case.
    model: str
        Name of the model either 'randomforest' or 'XGboost'.
    param: dict
        Dictionary with parameters names (str) as keys and distributions to try.
    n_iter: int
        Number of parameter settings that are sampled.
        n_iter trades off runtime vs quality of the solution.
    cv: int
        Number of folds for cross-validation.

    Return
    -----------------------
    grid: RandomSearchCV obj
        The fitted RandomSearchCV object that stores the training information and the best model.
    """
    if model == "randomforest":
        model = RandomForestRegressor(random_state=10)
    elif model == "XGboost":
        model = xgboost.XGBRegressor(tree_method="hist")
    else:
        raise ValueError("""model need to be either 'randomforest' or 'XGboost'!""")
    param = param

    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=param,
        n_iter=10,
        scoring=["neg_mean_absolute_error", "neg_mean_squared_error", "r2"],
        refit="r2",
        cv=5,
    )
    grid.fit(X, y["CQ"])

    return grid


def grid_performance(grid):
    """
    Print the performance of the best model from the RandomizedSearchCV
    using three metrics: r2, RMSE and MAE.

    Parameters
    -------------------
    grid: RandomSearchCV obj
        The fitted RandomSearchCV object that stores the training information and the best model.
    """
    train_r2 = np.sort(grid.cv_results_["mean_test_r2"])[-1]

    train_RMSE = math.sqrt(
        -np.sort(grid.cv_results_["mean_test_neg_mean_squared_error"])[-1]
    )

    train_MAE = -np.sort(grid.cv_results_["mean_test_neg_mean_absolute_error"])[-1]

    print(
        "training score: R2 = {}, RMSE = {}, MAE = {}".format(
            train_r2, train_RMSE, train_MAE
        )
    )
    print("Best estimator:", grid.best_estimator_)


def _print_test_results(table):
    """
    Helper function that print the test metrics based on the tabel.

    Parameters
    -----------------------
    table: dict
        Dictionary that stores y predicted by the model and y predicted by VASP.
    """
    test_r2 = r2_score(table["VASP_CQ"], table["md_CQ"])
    test_RMSE = math.sqrt(mean_squared_error(table["VASP_CQ"], table["md_CQ"]))
    test_MAE = mean_absolute_error(table["VASP_CQ"], table["md_CQ"])
    print(
        "test scores: R2 = {}, RMSE = {}, MAE = {}".format(test_r2, test_RMSE, test_MAE)
    )


def grid_test(X_test, y_test, grid, plot=True, is_O=False):
    """
    Return the evaluation metrics of the gird over the test data set.
    Also can plot the resultant correlation plot.

    Parameters
    ---------------------------
    X_test: array like
        Test data.
    y_test: array like
        Test labels.
    grid: RandomSearchCV obj
        The fitted RandomSearchCV object that stores the
        training information and the best model.
    plot: boo
        Plot the correlation between y_test and predicted y
        if plot==True.
    is_O: boo
        If is_O==True, separately plot points for pure oxygen sites
        and sites with other atomic types. Not useful when plot==False.
    """
    y_rf = pd.Series(grid.predict(X_test))
    y_test.reset_index(drop=True, inplace=True)

    test_result = pd.concat([y_test, y_rf], axis=1)
    test_result.rename(columns={"CQ": "VASP_CQ", 0: "md_CQ"}, inplace=True)

    _print_test_results(test_result)

    if plot:
        if is_O:
            sns.set(font_scale=1.5)
            plot = sns.lmplot(
                x="VASP_CQ",
                y="md_CQ",
                data=test_result,
                hue="is_O",
                height=6,
                aspect=5 / 4,
            )
            plot.set(
                xlabel="VASP calculated CQ (MHz)", ylabel="Model predicted CQ (MHz)"
            )
            plt.show()

        else:
            reg_plot(
                test_result["VASP_CQ"],
                test_result["md_CQ"],
                "VASP calculated CQ (MHz)",
                "Random Forest predicted CQ (MHz)",
            )
