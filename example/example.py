# -*- coding: utf-8 -*-
import json
import sys

import pandas as pd

sys.path.append("../")
from src.predict import predict
from src.model import _print_test_results

with open("../example/example_data.json", "r") as file:
    data = json.load(file)
    print("length of file is {}".format(len(data)))

pre = predict("xgboost.pkl", "struc+ele", data)
y_vasp, y_pre = pre.model_predict()

print("\n")
print("The model's prediction perfromance of CQ values for aluminum sites are:\n")
test_result = pd.concat([pd.Series(y_vasp), pd.Series(y_pre)], axis=1)
test_result.rename(columns={0: "VASP_CQ", 1: "md_CQ"}, inplace=True)
_print_test_results(test_result)
