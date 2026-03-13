import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/preprocessed_dataset.csv") # Loading the preprocessed dataset

# Summary
df.shape
df.info()
df.describe()

# Check target variable distribution
df['Default'].value_counts()
sns.countplot(x='Default', data=df)
plt.title("Distribution of Loan Defaults")
plt.show()

# Distribution of important numerical feaetures :
# Income
sns.histplot(df['Income'], bins=50, kde=True)
plt.title("Income Distribution")
plt.show()

# Loan amount
sns.histplot(df['LoanAmount'], bins=50, kde=True)
plt.title("Loan Amount Distribution")
plt.show()

# Interest rate
sns.histplot(df['InterestRate'], bins=50, kde=True)
plt.title("Interest Rate Distribution")
plt.show()

# DTI ratio
sns.histplot(df['DTIRatio'], bins=50, kde=True)
plt.title("Debt-to-Income Ratio Distribution")
plt.show()

# Income vs Default
sns.boxplot(x='Default', y='Income', data=df)
plt.title("Income vs Loan Default")
plt.show()

# Loan amount vs Default
sns.boxplot(x='Default', y='LoanAmount', data=df)
plt.title("Loan Amount vs Default")
plt.show()

# Credit band vs Default
sns.countplot(x='CreditBand', hue='Default', data=df)
plt.title("Credit Band vs Default")
plt.show()

# Age group vs Default
sns.countplot(x='AgeGroup', hue='Default', data=df)
plt.title("Age Group vs Default")
plt.show()

# Loan purpose vs Default

# Business loan
sns.barplot(x='LoanPurpose_Business', y='Default', data=df)
plt.title("Business Loan Purpose vs Default")
plt.show()

# Home loan
sns.barplot(x='LoanPurpose_Home', y='Default', data=df)
plt.title("Home Loan Purpose vs Default")
plt.show()

# Education loan
sns.barplot(x='LoanPurpose_Education', y='Default', data=df)
plt.title("Education Loan Purpose vs Default")
plt.show()

# Other
sns.barplot(x='LoanPurpose_Other', y='Default', data=df)
plt.title("Other Loan Purpose vs Default")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Default rate by feature

# Credit band
print(df.groupby('CreditBand')['Default'].mean())
# Plot
sns.barplot(x='CreditBand', y='Default', data=df)
plt.title("Default Rate by Credit Band")
plt.show()

# Age group
print(df.groupby('AgeGroup')['Default'].mean())
# Plot
sns.barplot(x='AgeGroup', y='Default', data=df)
plt.title("Default Rate by Age Group")
plt.show()

