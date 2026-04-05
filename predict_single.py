import pandas as pd
import numpy as np
import joblib

# 1. LOAD ASSETS
# Loading assets created in train_model.py
model = joblib.load("models/random_forest.pkl") 
scaler = joblib.load("models/scaler.pkl") 

# 2. DEFINE NEW CUSTOMER DATA
new_customer = {
    'Age': 35,
    'Income': 75000,
    'LoanAmount': 15000,
    'CreditScore': 720,
    'MonthsEmployed': 48,
    'InterestRate': 5.5,
    'DTIRatio': 0.3,
    'Education': "Master's",
    'HasMortgage': 'Yes',
    'HasDependents': 'No',
    'HasCoSigner': 'No',
    'EmploymentType': 'Full-time',
    'MaritalStatus': 'Married',
    'LoanPurpose': 'Home'
}

def predict_default(data):
    # Convert single dictionary to DataFrame
    df = pd.DataFrame([data])
    
    # 3. REPLICATE PREPROCESSING STEPS
    # Binary Mapping
    binary_map = {'Yes': 1, 'No': 0}
    for col in ['HasMortgage', 'HasDependents', 'HasCoSigner']:
        df[col] = df[col].map(binary_map)
        
    # Ordinal Mapping
    edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    df['Education'] = df['Education'].map(edu_map)
    
    # Discretization (Binning)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 100], labels=[0, 1, 2, 3])
    df['CreditBand'] = pd.cut(df['CreditScore'], bins=[300, 580, 670, 740, 850], labels=[0, 1, 2, 3])
    
    # Nominal Encoding (One-Hot)
    # Using drop_first=True to match preprocessing.py
    df = pd.get_dummies(df, columns=['EmploymentType', 'MaritalStatus', 'LoanPurpose'], drop_first=True)
    
    # 4. ALIGN COLUMNS (The Fix)
    # Get the exact columns the scaler and model were trained on
    training_cols = scaler.feature_names_in_
    
    # Add any missing columns with 0 (like the other one-hot categories)
    for col in training_cols:
        if col not in df.columns:
            df[col] = 0
            
    # Reorder columns to match the training set exactly
    df = df[training_cols]
    
    # 5. SCALE AND PREDICT
    # Scale the entire row because scaler.pkl was fitted on X_train
    df_scaled = scaler.transform(df)
    
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    
    return prediction, probability

# Run the prediction
pred, prob = predict_default(new_customer)

print("-" * 30)
print(f"RESULTS FOR CUSTOMER")
print("-" * 30)
status = "⚠️  HIGH RISK (Default Likely)" if pred == 1 else "✅ LOW RISK (Safe)"
print(f"Prediction: {status}")
print(f"Probability of Default: {prob:.2%}")
print("-" * 30)