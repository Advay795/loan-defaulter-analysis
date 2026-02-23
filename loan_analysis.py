import numpy as np
import pandas as pd

df = pd.read_csv("datasets/dataset-1.csv")

print(df.head())
print(df.info())
print(df.describe())