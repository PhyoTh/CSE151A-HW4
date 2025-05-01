# Mystery Dataset Analysis: Finding Relevant Features

# 1. Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 2. Load and Preprocess the Data
# Load the mystery dataset
data = np.loadtxt('mystery.dat', delimiter=',')

# Separate features and target variable
X = data[:, :-1]
y = data[:, -1]

print(f"Dataset shape: X: {X.shape}, y: {y.shape}")

# 3. Split the Data and Standardize Features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Lasso Regression with Cross-Validation
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print(f"\nOptimal alpha value: {lasso_cv.alpha_}")

# Train Lasso with best alpha
lasso_model = Lasso(alpha=lasso_cv.alpha_, random_state=42, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = lasso_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")

# 5. Identify Selected Features from Lasso
lasso_coef = lasso_model.coef_
abs_coef = np.abs(lasso_coef)
threshold = 1e-10
selected_features_lasso = np.where(abs_coef > threshold)[0]

print(f"\nNumber of features selected by Lasso (threshold={threshold}): {len(selected_features_lasso)}")
print(f"Selected feature indices (1-indexed): {selected_features_lasso + 1}")

# Format Lasso selected features as a DataFrame
lasso_features_df = pd.DataFrame({
    'Feature Index': selected_features_lasso + 1,
    'Coefficient': lasso_coef[selected_features_lasso]
})
print("\nLasso Selected Features:")
print(lasso_features_df)

# 6. Recursive Feature Elimination (RFE)
base_model = LinearRegression()
rfe = RFE(estimator=base_model, n_features_to_select=10, step=1)
rfe.fit(X_train_scaled, y_train)
selected_features_rfe = np.where(rfe.support_)[0]

print(f"\nNumber of features selected by RFE: {len(selected_features_rfe)}")
print(f"Selected feature indices (1-indexed): {selected_features_rfe + 1}")

# 7. Train Model on RFE Selected Features
model = LinearRegression()
model.fit(X_train_scaled[:, selected_features_rfe], y_train)

# Evaluate
y_pred_rfe = model.predict(X_test_scaled[:, selected_features_rfe])
mse_rfe = mean_squared_error(y_test, y_pred_rfe)

# Format RFE selected features as a DataFrame with coefficients
rfe_features_df = pd.DataFrame({
    'Feature Index': selected_features_rfe + 1,
    'Coefficient': model.coef_
})
print(rfe_features_df)

print(f"\nMean Squared Error with 10 selected features: {mse_rfe:.4f}")