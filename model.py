
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the real dataset
df = pd.read_csv("student-mat.csv", sep=";")

print("\n========== DATASET OVERVIEW ==========")
print("Dataset shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())

print("\n========== DATA QUALITY ==========")
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# 2. Explore final grades
print("\n========== FINAL GRADE ANALYSIS ==========")
print(df["G3"].describe())

plt.figure(figsize=(8, 5))
plt.hist(df["G3"], bins=11, edgecolor="black")
plt.title("Distribution of Final Student Grades")
plt.xlabel("Final Grade (0–20)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# 3. Prepare features and target
# G1 and G2 are earlier-period grades; G3 is the final grade.
# We exclude G1 and G2 to test prediction without earlier grades.
features = ["studytime", "failures", "absences", "Medu", "Fedu",
            "traveltime", "freetime", "goout", "health"]

X = df[features]
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train and compare two models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42)
}

results = {}
predictions = {}

print("\n========== MODEL COMPARISON ==========")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "MSE": mse, "R2": r2}
    predictions[name] = y_pred

    print(f"\n{name}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")

# 5. Plot actual vs predicted grades for Linear Regression
plt.figure(figsize=(7, 5))
plt.scatter(y_test, predictions["Linear Regression"], alpha=0.7)
plt.plot([0, 20], [0, 20], "r--")
plt.title("Actual vs Predicted Final Grades")
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.tight_layout()
plt.show()

print("\nAnalysis complete.")
