import numpy as np
import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "results" / "final_predictions.csv"

df = pd.read_csv(file_path)

print(df.head())
print(df.info())
print(df.describe())
