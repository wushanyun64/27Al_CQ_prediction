# -*- coding: utf-8 -*-
import json
import sys

sys.path.append("../")
from src.predict import predict

with open("example_data.json", "r") as file:
    data = json.load(file)
    print("length of file is {}".format(len(data)))

pre = predict("struc+ele.pkl", "struc+ele", data)
y = pre.model_predict()

print("\n")
print("The model's prediction of CQ values for aluminum sites are:\n")
print(y)
