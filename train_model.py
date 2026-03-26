import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train = pd.read_csv("datasets/train_features.csv")
X_test = pd.read_csv("datasets/test_features.csv")

y_train = pd.read_csv("datasets/train_target.csv")
y_test = pd.read_csv("datasets/test_target.csv")

#handling missing values 
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean()) 

for col in X_train.select_dtypes(include='object').columns:
    mode = X_train[col].mode()[0]
    X_train[col].fillna(mode, inplace=True)
    X_test[col].fillna(mode, inplace=True)

y_train = y_train.squeeze()
y_test = y_test.squeeze()

#feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#logistic regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

#decision tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

#random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))