
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor 

st.set_page_config(
    page_title="Social Media Engagement Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Social Media Engagement Prediction")
st.write("Analyze social media posts and compare engagement across platforms.")

# Locate the dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "sample_train.csv"

if not DATA_PATH.exists():
    st.error("train.csv not found. Put it inside the dataset folder.")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.write("Dataset columns:", df.columns.tolist())

# Calculate engagement score if it is not already available
if "engagement_score" not in df.columns:
    if "likes" in df.columns and "comments" in df.columns:
        df["engagement_score"] = df["likes"] + df["comments"]
    else:
        st.error("Dataset must contain likes and comments columns.")
        st.stop()

# Sidebar filters
st.sidebar.header("Dashboard Filters")

if "platform" in df.columns:
    platforms = sorted(df["platform"].dropna().unique())
    selected = st.sidebar.multiselect(
        "Choose platform",
        platforms,
        default=platforms
    )
    filtered = df[df["platform"].isin(selected)]
else:
    filtered = df.copy()

if filtered.empty:
    st.warning("Select at least one platform.")
    st.stop()

# Summary cards
c1, c2, c3 = st.columns(3)
c1.metric("Total Posts", f"{len(filtered):,}")
c2.metric(
    "Average Engagement",
    f"{filtered['engagement_score'].mean():,.2f}"
)
c3.metric(
    "Highest Engagement",
    f"{filtered['engagement_score'].max():,.0f}"
)

st.divider()

# Platform comparison
if "platform" in filtered.columns:
    st.subheader("Average Engagement by Platform")
    platform_avg = filtered.groupby("platform")[
        "engagement_score"
    ].mean().sort_values(ascending=False)

    st.bar_chart(platform_avg)

# Content type analysis
if "content_type" in filtered.columns:
    st.subheader("Posts by Content Type")
    content_counts = filtered["content_type"].value_counts()
    st.bar_chart(content_counts)

# Likes and comments
if "likes" in filtered.columns and "comments" in filtered.columns:
    st.subheader("Likes and Comments Analysis")
    st.bar_chart(filtered[["likes", "comments"]].mean())

# Dataset preview
with st.expander("View Dataset"):
    st.dataframe(filtered, use_container_width=True)

st.caption("Social Media Engagement Analysis | Mini Project")


# Prediction section
st.divider()
st.header("Predict Social Media Engagement")

st.write("Enter the details of a post to estimate its engagement.")

platform_options = (
    sorted(df["platform"].dropna().unique())
    if "platform" in df.columns
    else ["instagram", "facebook", "twitter"]
)

selected_platform = st.selectbox(
    "Select Platform",
    platform_options
)

if "content_type" in df.columns:
    content_options = sorted(df["content_type"].dropna().unique())
else:
    content_options = ["image", "video", "text"]

selected_content = st.selectbox(
    "Select Content Type",
    content_options
)

if "caption_length" in df.columns:
    caption_length = st.number_input(
        "Caption Length",
        min_value=0,
        value=120
    )
else:
    caption_length = 120



    try:
        prediction_data = df.copy()

        if "caption_length" not in prediction_data.columns:
            prediction_data["caption_length"] = 120
        if "post_day" not in prediction_data.columns:
            prediction_data["post_day"] = 4
        if "post_month" not in prediction_data.columns:
            prediction_data["post_month"] = 10

        features = [
            "platform",
            "content_type",
            "caption_length",
            "post_day",
            "post_month"
        ]

        X = prediction_data[features]
        y = prediction_data["likes"]

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

        model = Pipeline([
            ("preprocess", preprocess),
            ("model", RandomForestRegressor(
                n_estimators=150,
                random_state=42
            ))
        ])

        model.fit(X, y)

        post = pd.DataFrame([{
            "platform": selected_platform,
            "content_type": selected_content,
            "caption_length": caption_length,
            "post_day": 4,
            "post_month": 10
        }])

        prediction = max(0, float(model.predict(post)[0]))

        st.success(f"Predicted Likes: {prediction:.0f}")
        st.caption("This is an estimated number of likes, not a guarantee.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

if st.button("Predict Engagement"):
    try:
        prediction_data = df.copy()

        # Create features from the real dataset columns
        prediction_data["caption_length"] = (
            prediction_data["caption"].fillna("").str.len()
        )

        dates = pd.to_datetime(
            prediction_data["date_posted"], errors="coerce"
        )

        prediction_data["post_day"] = dates.dt.day
        prediction_data["post_month"] = dates.dt.month

        prediction_data["post_day"] = (
            prediction_data["post_day"].fillna(1)
        )
        prediction_data["post_month"] = (
            prediction_data["post_month"].fillna(1)
        )

        features = [
            "platform",
            "content_type",
            "caption_length",
            "post_day",
            "post_month"
        ]

        prediction_data = prediction_data.dropna(
            subset=["likes", "platform", "content_type"]
        )

        X = prediction_data[features]
        y = prediction_data["likes"]

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

        model = Pipeline([
            ("preprocess", preprocess),
            ("model", RandomForestRegressor(
                n_estimators=150,
                random_state=42
            ))
        ])

        model.fit(X, y)

        caption = st.text_area(
            "Enter a sample caption",
            "Check out our latest update!"
        )

        post = pd.DataFrame([{
            "platform": selected_platform,
            "content_type": selected_content,
            "caption_length": len(caption),
            "post_day": pd.Timestamp.today().day,
            "post_month": pd.Timestamp.today().month
        }])

        prediction = max(0, float(model.predict(post)[0]))

        st.success(f"Predicted Likes: {prediction:.0f}")
        st.caption(
            "This is a model estimate based on historical dataset patterns, "
            "not a guarantee of actual likes."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
