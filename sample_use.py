import skxplore as skx
import pandas as pd

# Test code for clustering
dataset = pd.read_csv('datasets/iris.csv')
skx.find_model(dataset, 1, "clustering", label="species")

# Test code for classification
dataset = pd.read_csv('datasets/balance-scale.data')
skx.find_model(dataset, 0.7, "classification", label="label", datatype="numerical", dim_reduction=True, components="auto", contains_negative=True, ensembling=True, priority="accuracy")

# Test code for regression
dataset = pd.read_csv('datasets/boston.csv')
skx.find_model(dataset, 0.8, "regression", label="medv", dim_reduction=False, components="auto", contains_negative=False, ensembling=False)