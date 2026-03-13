import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("datasets/primary_dataset.csv")

df.drop(columns=['LoanID'], inplace=True) # Dropping useless columns

df['Default'].value_counts(normalize=True) # For checking target balance

# Converting binary categories
binary_map = {'Yes':1, 'No':0}
df['HasMortgage'] = df['HasMortgage'].map(binary_map)
df['HasDependents'] = df['HasDependents'].map(binary_map)
df['HasCoSigner'] = df['HasCoSigner'].map(binary_map)

# Ordinal encoding
edu_map = {
 "High School":0,
 "Bachelor's":1,
 "Master's":2,
 "PhD":3
}
df['Education'] = df['Education'].map(edu_map)

# Nominal encoding
df = pd.get_dummies(df, columns=[
 'EmploymentType',
 'MaritalStatus',
 'LoanPurpose'
], drop_first=True)

# Discretization
df['AgeGroup'] = pd.cut(df['Age'], bins=[18,30,45,60,100], labels=[0,1,2,3])
df['CreditBand'] = pd.cut(df['CreditScore'], bins=[300,580,670,740,850], labels=[0,1,2,3])

# Feature scaling
scaler = StandardScaler()
num_cols = ['Income','LoanAmount','CreditScore',
            'MonthsEmployed','InterestRate','DTIRatio']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Train test split
X = df.drop('Default', axis=1)
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Store the pre-processed data
df.to_csv("datasets/preprocessed_dataset.csv", index=False)

# Store split data
X_train.to_csv("datasets/train_features.csv", index=False)
X_test.to_csv("datasets/test_features.csv", index=False)
y_train.to_csv("datasets/train_target.csv", index=False)
y_test.to_csv("datasets/test_target.csv", index=False)
