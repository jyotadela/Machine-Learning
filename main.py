import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Joshna\Downloads\StudentsPerformance.csv")

# Display first 5 rows
print(df.head())

# Dataset information
print(df.info())

# Check missing values
print(df.isnull().sum())

# Statistical summary
print(df.describe())

plt.figure(figsize=(8,5))
sns.histplot(df["math score"], bins=20)
plt.title("Math Score Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="gender", y="math score", data=df)
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x="test preparation course", data=df)
plt.show()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

for col in categorical:
    df[col] = le.fit_transform(df[col])
X = df.drop("math score", axis=1)
y = df["math score"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

print("Decision Tree R2:", dt.score(X_test, y_test))
print("Random Forest R2:", rf.score(X_test, y_test))
print("Linear Regression R2:", model.score(X_test, y_test))
import joblib

joblib.dump(rf, "student_performance_model.pkl")

print("Model Saved Successfully")