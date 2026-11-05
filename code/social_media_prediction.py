"""
Social Media Engagement Prediction Using Statistical Machine Learning
Run from the project root: python code/social_media_prediction.py
Input: dataset/train.csv (your Kaggle data) OR dataset/sample_train.csv (included sample)
Target: likes. Predictors use post metadata only to avoid using comments/likes as inputs.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
GRAPH_DIR = ROOT / "graphs"
RESULT_DIR = ROOT / "results"
GRAPH_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# Prefer the user's Kaggle file if it exists; otherwise use included sample data.
real_data = DATA_DIR / "train.csv"
sample_data = DATA_DIR / "sample_train.csv"
data_path = real_data if real_data.exists() else sample_data
df = pd.read_csv(data_path)
print(f"Loaded dataset: {data_path.name}")
print("\\nFirst five rows:")
print(df.head())
print(f"\\nDataset shape: {df.shape}")
print("\\nColumn names:", df.columns.tolist())
print("\\nMissing values:")
print(df.isnull().sum())
print("\\nDuplicate rows:", df.duplicated().sum())
print("\\nNumeric statistical summary:")
print(df.describe(include="number"))

# Basic cleaning
df = df.drop_duplicates().copy()
for col in ["platform", "content_type", "caption", "date_posted"]:
    if col not in df.columns:
        df[col] = "unknown" if col != "caption" else ""
df["platform"] = df["platform"].fillna("unknown").astype(str)
df["content_type"] = df["content_type"].fillna("unknown").astype(str)
df["caption"] = df["caption"].fillna("").astype(str)
df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
df["caption_length"] = df["caption"].str.len()
df["post_day"] = df["date_posted"].dt.dayofweek.fillna(-1).astype(int)
df["post_month"] = df["date_posted"].dt.month.fillna(0).astype(int)

# EDA graph 1: number of posts per platform
counts = df["platform"].value_counts()
plt.figure(figsize=(8,5))
counts.plot(kind="bar")
plt.title("Number of Posts by Platform")
plt.xlabel("Platform")
plt.ylabel("Number of Posts")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "posts_by_platform.png", dpi=150)
plt.close()

# EDA graph 2: average likes by platform
if "likes" not in df.columns:
    raise ValueError("The dataset must contain a 'likes' column to run this project.")
df["likes"] = pd.to_numeric(df["likes"], errors="coerce")
df = df.dropna(subset=["likes"])
df["likes"] = df["likes"].clip(lower=0)
avg_likes = df.groupby("platform")["likes"].mean().sort_values(ascending=False)
plt.figure(figsize=(8,5))
avg_likes.plot(kind="bar")
plt.title("Average Likes by Platform")
plt.xlabel("Platform")
plt.ylabel("Average Likes")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "average_likes_by_platform.png", dpi=150)
plt.close()

# Statistical correlation graph: likes vs caption length
plt.figure(figsize=(8,5))
plt.scatter(df["caption_length"], df["likes"], alpha=0.5)
plt.title("Caption Length vs Likes")
plt.xlabel("Caption Length (characters)")
plt.ylabel("Likes")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "caption_length_vs_likes.png", dpi=150)
plt.close()

# Predict likes from metadata only (do not use likes/comments as predictors).
features = ["platform", "content_type", "caption_length", "post_day", "post_month"]
X = df[features]
y = df["likes"]

categorical = ["platform", "content_type"]
numeric = ["caption_length", "post_day", "post_month"]
preprocess = ColumnTransformer([
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=2)
}
rows = []
for name, model in models.items():
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    rows.append({"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})
    if name == "Random Forest":
        # Example prediction for a new post using metadata only.
        example = pd.DataFrame([{
            "platform": "instagram",
            "content_type": "video",
            "caption_length": 120,
            "post_day": 4,
            "post_month": 10
        }])
        example_prediction = float(pipeline.predict(example)[0])
        print("\\nExample new-post metadata:", example.iloc[0].to_dict())
        print(f"Predicted likes (Random Forest): {max(0, example_prediction):.1f}")

results = pd.DataFrame(rows).sort_values("RMSE")
results.to_csv(RESULT_DIR / "model_comparison.csv", index=False)
print("\\nModel comparison (lower MAE/RMSE is better; R2 closer to 1 is better):")
print(results.to_string(index=False))

plt.figure(figsize=(8,5))
plt.bar(results["Model"], results["RMSE"])
plt.title("Model Comparison by RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "model_comparison_rmse.png", dpi=150)
plt.close()

print(f"\\nGraphs saved in: {GRAPH_DIR}")
print(f"Model results saved in: {RESULT_DIR / 'model_comparison.csv'}")
print("\\nProject completed successfully.")
