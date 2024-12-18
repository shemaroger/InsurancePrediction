import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Task 1: Data Preparation
# Load dataset (assuming it's in CSV format)
df = pd.read_csv("D:\\Project\\FinalExam\\data\\insurance_prediction\\insurance.csv")  # Ensure correct file path

# Clean the data: Handle missing values and duplicates
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns  # Select numerical columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())  # Replace missing values with column mean
df.drop_duplicates(inplace=True)  # Remove duplicate rows

# Encode categorical columns (sex, smoker, region)
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

# Feature scaling for numerical columns
scaler = StandardScaler()
df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])

# Task 2: Exploratory Data Analysis
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Boxplot to analyze charges by smoker status
plt.figure(figsize=(8, 6))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Charges by Smoker Status')
plt.xlabel('Smoker (0 = No, 1 = Yes)')
plt.ylabel('Charges')
plt.show()

# Pairplot to visualize distributions and relationships
sns.pairplot(df, vars=['age', 'bmi', 'charges'], hue='smoker')
plt.show()
# 1. Summary statistics
print("\nSummary Statistics: ")
print(df.describe())

# Task 3: Predictive Modeling
# Define features (X) and target (y)
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save the trained model and scaler
joblib.dump(model, 'insurance_model.joblib')
joblib.dump(scaler, 'scaler.joblib')