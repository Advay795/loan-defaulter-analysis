import pandas as pd
df = pd.read_csv("Loan Default Prediction Dataset Copy export 2026-02-09 05-43-52.csv")
print("cols with missing values")
print(df.isnull().sum())
print("%-missing values")
print(((df.isnull().sum())/len(df))*100)
print("summary of the dataset: ")
print(df.info())
print("target imbalance")
print(df['loan_status'].value_counts())
print("%-target-imbalance")
print(df['loan_status'].value_counts(normalize=True) * 100)
summary = pd.DataFrame({
    'Data Type': df.dtypes,
    'Missing Values': df.isnull().sum(),
    '% Missing': (df.isnull().sum()/len(df))*100,
    'Unique Values': df.nunique()
})
print(summary)

