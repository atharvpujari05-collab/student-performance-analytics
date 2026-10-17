
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Analytics")
st.write(
    "Explore student data, compare machine learning models, "
    "and estimate final mathematics grades."
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("student-mat.csv", sep=";")

df = load_data()

# Features match model.py
features = [
    "studytime", "failures", "absences", "Medu", "Fedu",
    "traveltime", "freetime", "goout", "health"
]
target = "G3"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train both models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=4, random_state=42
    )
}

results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    trained_models[name] = model
    results[name] = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "R2": r2_score(y_test, predictions)
    }

# Dashboard metrics
st.header("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Student Records", len(df))
c2.metric("Dataset Features", len(df.columns))
c3.metric("Average Final Grade", f"{df['G3'].mean():.2f}/20")
c4.metric("Average Absences", f"{df['absences'].mean():.1f}")

st.divider()

# Dataset preview
with st.expander("View Dataset Preview"):
    st.dataframe(df.head(20), use_container_width=True)

# Visual analysis
st.header("Exploratory Data Analysis")

left, right = st.columns(2)

with left:
    st.subheader("Final Grade Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["G3"], bins=11, edgecolor="black")
    ax.set_xlabel("Final Grade (0–20)")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Final Grades")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Study Time vs Final Grade")
    fig, ax = plt.subplots()
    ax.scatter(df["studytime"], df["G3"], alpha=0.6)
    ax.set_xlabel("Study Time Category (1–4)")
    ax.set_ylabel("Final Grade (0–20)")
    ax.set_title("Study Time and Final Grade")
    st.pyplot(fig)
    plt.close(fig)

left, right = st.columns(2)

with left:
    st.subheader("Absences vs Final Grade")
    fig, ax = plt.subplots()
    ax.scatter(df["absences"], df["G3"], alpha=0.6)
    ax.set_xlabel("Number of Absences")
    ax.set_ylabel("Final Grade (0–20)")
    ax.set_title("Absences and Final Grade")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Feature Correlation")
    corr = df[features + [target]].corr()
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# Model comparison
st.header("Machine Learning Model Comparison")

comparison = pd.DataFrame(results).T
st.dataframe(comparison.round(3), use_container_width=True)

st.caption(
    "Lower MAE and MSE indicate smaller prediction errors. "
    "R² describes how much variation the model explains; "
    "a higher value is generally better."
)

# Prediction interface
https://student-performance-analytics-lhzs97vbdgktkuuhz6pefd.streamlit.app/
st.divider()
st.caption(
    "Dataset: UCI Student Performance | "
    "Models: Linear Regression and Decision Tree"
)

