# 📌 Feature Engineering for Machine Learning

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ✅ Load Dataset (Housing Dataset)
url = "https://raw.githubusercontent.com/Phionanamugga/teaching/refs/heads/main/datasets/housing.csv"
df = pd.read_csv(url)

# ✅ Display Initial Data
print("Original Data:\n", df.head())

# 🔹 Handle Missing Values 🔹
imputer = SimpleImputer(strategy="median")
df_numeric = df.select_dtypes(include=[np.number])  # Select numerical columns
df[df_numeric.columns] = imputer.fit_transform(df_numeric)

# 🔹 Encode Categorical Features 🔹
categorical_features = df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_categorical = encoder.fit_transform(df[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)

# Convert to DataFrame
df_encoded = pd.DataFrame(encoded_categorical, columns=encoded_feature_names, index=df.index)
df = df.drop(columns=categorical_features).join(df_encoded)

# 🔹 Feature Scaling 🔹
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 🔹 Polynomial Features 🔹
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
df_poly = pd.DataFrame(poly.fit_transform(df_scaled), columns=poly.get_feature_names_out(df.columns))

# ✅ Final Transformed Data
print("\nTransformed Data:\n", df_poly.head())

# ✅ Save Processed Data
df_poly.to_csv("processed_housing_data.csv", index=False)




