# -*- coding: utf-8 -*-
"""
Student Service Request Monitoring - Streamlit App
Converted from: BSIS_3B _VILLATURA_MARK JOHN_L._.ipynb

Run with:  streamlit run app.py
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Ellipse, Rectangle
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Student Service Request Monitoring",
    page_icon="📊",
    layout="wide",
)

CATEGORICAL_PREVIEW = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Required_Documents_Complete", "Status",
]

CATEGORY_COLUMNS = [
    "Service_Type", "Requester_Type", "Requested_Channel",
    "Priority", "Assigned_Office",
    "Required_Documents_Complete", "Status", "Reopened",
]

NUMERIC_COLUMNS = ["Queue_Length", "Processing_Days", "SLA_Days", "Satisfaction_Score"]

MODEL_CATEGORICAL = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Required_Documents_Complete", "Submission_Weekday",
]
MODEL_NUMERICAL = ["Queue_Length", "SLA_Days"]


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame):
    """Clean the raw dataset and engineer features. Returns (clean, model_df, log)."""
    clean = df.copy()
    log = [f"Rows before cleaning: {len(clean)}"]

    # 1. Standardize column names
    clean.columns = clean.columns.str.strip()

    # 2. Duplicate Request_ID
    clean["Request_ID"] = clean["Request_ID"].astype("string").str.strip()
    dup = int(clean["Request_ID"].duplicated(keep="first").sum())
    log.append(f"Duplicate Request_ID rows removed: {dup}")
    clean = clean.loc[~clean["Request_ID"].duplicated(keep="first")].copy()

    # 3. Standardize categorical text
    for col in CATEGORY_COLUMNS:
        if col in clean.columns:
            old = clean[col].astype("string")
            clean[col] = (
                old.str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
                .str.title()
            )
            changes = int((old.fillna("") != clean[col].fillna("")).sum())
            log.append(f"{col}: cells changed during text standardization = {changes}")

    # 4. Dates
    clean["Submission_Date"] = pd.to_datetime(clean["Submission_Date"], errors="coerce")
    log.append(f"Invalid or missing submission dates: {int(clean['Submission_Date'].isna().sum())}")

    # 5. Numeric columns
    for col in NUMERIC_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        neg = int((clean[col] < 0).sum())
        if neg > 0:
            clean.loc[clean[col] < 0, col] = np.nan
        log.append(f"{col}: negative values changed to missing = {neg}")

    # 6. Range check: satisfaction 1-5
    bad_sat = clean["Satisfaction_Score"].notna() & ~clean["Satisfaction_Score"].between(1, 5)
    log.append(f"Invalid satisfaction ratings: {int(bad_sat.sum())}")
    clean.loc[bad_sat, "Satisfaction_Score"] = np.nan

    log.append(f"Rows after cleaning: {len(clean)}")

    # Feature engineering
    clean["Submission_Weekday"] = clean["Submission_Date"].dt.day_name()

    clean["Queue_Category"] = pd.Series(pd.NA, index=clean.index, dtype="string")
    clean.loc[clean["Queue_Length"].between(0, 5), "Queue_Category"] = "Low"
    clean.loc[clean["Queue_Length"].between(6, 15), "Queue_Category"] = "Medium"
    clean.loc[clean["Queue_Length"] > 15, "Queue_Category"] = "High"

    eligible = (
        clean["Status"].eq("Completed")
        & clean["Processing_Days"].notna()
        & clean["SLA_Days"].notna()
    )
    clean["SLA_Breached"] = pd.Series(pd.NA, index=clean.index, dtype="Int64")
    clean.loc[eligible, "SLA_Breached"] = (
        clean.loc[eligible, "Processing_Days"] > clean.loc[eligible, "SLA_Days"]
    ).astype(int)

    model_df = clean.loc[eligible & clean["SLA_Breached"].notna()].copy()
    return clean, model_df, log


def breach_table(model_df: pd.DataFrame, by: str) -> pd.DataFrame:
    out = (
        model_df.groupby(by, observed=True)
        .agg(
            Total_Requests=("SLA_Breached", "size"),
            Breaches=("SLA_Breached", "sum"),
            Breach_Rate=("SLA_Breached", "mean"),
        )
    )
    out["Breach_Rate_Percent"] = (out["Breach_Rate"] * 100).round(2)
    return out


@st.cache_resource(show_spinner="Training Decision Tree...")
def train_model(model_df: pd.DataFrame, max_depth: int, min_leaf: int, test_size: float, seed: int):
    features = MODEL_CATEGORICAL + MODEL_NUMERICAL
    data = model_df.dropna(subset=["SLA_Breached"]).copy()
    X = data[features]
    y = data["SLA_Breached"].astype(int)

    if y.nunique() < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                MODEL_CATEGORICAL,
            ),
            (
                "numerical",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                MODEL_NUMERICAL,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_leaf, random_state=seed
        )),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (breach)": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall (breach)": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1-score (breach)": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
    }

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    b_pred = baseline.predict(X_test)
    baseline_metrics = {
        "Accuracy": accuracy_score(y_test, b_pred),
        "Precision (breach)": precision_score(y_test, b_pred, pos_label=1, zero_division=0),
        "Recall (breach)": recall_score(y_test, b_pred, pos_label=1, zero_division=0),
        "F1-score (breach)": f1_score(y_test, b_pred, pos_label=1, zero_division=0),
    }

    report = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["No Breach", "Breach"], zero_division=0, output_dict=True,
    )

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame(
        {"Encoded_Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    return {
        "y": y,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "report": pd.DataFrame(report).T,
        "cm": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "importance_df": importance_df,
    }


def original_feature(encoded_name: str) -> str:
    name = encoded_name.replace("categorical__", "").replace("numerical__", "")
    for prefix in MODEL_CATEGORICAL:
        if name.startswith(prefix + "_"):
            return prefix
    return name


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------
def make_use_case_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.add_patch(Rectangle((3.0, 0.7), 6.2, 7.6, fill=False, linewidth=2))
    ax.text(6.1, 8.0, "Student Service Request Monitoring System",
            ha="center", va="center", fontsize=12, weight="bold")

    use_cases = [
        (6.1, 6.7, "View Service Volume"),
        (6.1, 5.4, "Review SLA Breach Rate"),
        (6.1, 4.1, "View Queue Risk"),
        (6.1, 2.8, "Review Flagged Requests"),
        (6.1, 1.5, "View Model Explanation"),
    ]
    for x, y, label in use_cases:
        ax.add_patch(Ellipse((x, y), 3.2, 0.8, facecolor="white",
                             edgecolor="black", linewidth=1.4))
        ax.text(x, y, label, ha="center", va="center", fontsize=9)

    def draw_actor(x, y, label):
        ax.add_patch(Ellipse((x, y + 0.65), 0.35, 0.35, fill=False))
        ax.plot([x, x], [y + 0.48, y - 0.25], color="black")
        ax.plot([x - 0.35, x + 0.35], [y + 0.2, y + 0.2], color="black")
        ax.plot([x, x - 0.3], [y - 0.25, y - 0.65], color="black")
        ax.plot([x, x + 0.3], [y - 0.25, y - 0.65], color="black")
        ax.text(x, y - 0.95, label, ha="center", va="top", fontsize=10)

    draw_actor(1.4, 5.0, "Administrator")
    draw_actor(10.7, 5.0, "Service Staff")

    for y in [6.7, 5.4, 4.1, 2.8, 1.5]:
        ax.plot([1.8, 4.5], [5.2, y], color="gray", linewidth=1)
    ax.plot([10.3, 7.7], [5.2, 2.8], color="gray", linewidth=1)
    ax.plot([10.3, 7.7], [5.2, 1.5], color="gray", linewidth=1)

    ax.set_title("UML Use Case Diagram — Analytics and Decision Support", pad=15)
    fig.tight_layout()
    return fig


def make_wireframe():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(6.5, 8.5, "STUDENT SERVICE REQUEST MONITORING DASHBOARD",
            ha="center", va="center", fontsize=14, weight="bold")

    cards = [
        (0.5, 6.7, 2.8, 1.2, "Total Requests"),
        (3.6, 6.7, 2.8, 1.2, "SLA Breach Rate"),
        (6.7, 6.7, 2.8, 1.2, "Open Requests"),
        (9.8, 6.7, 2.8, 1.2, "Flagged Requests"),
    ]
    for x, y, w, h, label in cards:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

    ax.add_patch(Rectangle((0.5, 3.7), 5.9, 2.3, fill=False, linewidth=1.5))
    ax.text(3.45, 4.85, "Chart: Requests by Service Type",
            ha="center", va="center", fontsize=11)
    ax.add_patch(Rectangle((6.7, 3.7), 5.9, 2.3, fill=False, linewidth=1.5))
    ax.text(9.65, 4.85, "Chart: SLA Breach Rate by Queue",
            ha="center", va="center", fontsize=11)

    ax.add_patch(Rectangle((0.5, 0.8), 12.1, 2.2, fill=False, linewidth=1.5))
    ax.text(6.55, 2.6, "Flagged Requests for Human Review",
            ha="center", va="center", fontsize=11, weight="bold")
    ax.text(6.55, 1.8, "Request ID | Service Type | Priority | Queue | Risk Level | Reason",
            ha="center", va="center", fontsize=9)
    ax.text(6.55, 1.2, "Select a request to view details and model explanation",
            ha="center", va="center", fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📊 Student Service Request Monitoring")
st.caption("Data cleaning, SLA breach analytics, and Decision Tree prediction.")

with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader("Upload the CSV file", type=["csv"])

    st.header("2. Model settings")
    max_depth = st.slider("Max tree depth", 1, 15, 5)
    min_leaf = st.slider("Min samples per leaf", 1, 50, 5)
    test_size = st.slider("Test set size", 0.10, 0.50, 0.20, step=0.05)
    seed = st.number_input("Random seed", value=42, step=1)

if uploaded is None:
    st.info("👈 Upload your student service requests CSV in the sidebar to begin.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

required = ["Request_ID", "Submission_Date", "Status"] + NUMERIC_COLUMNS
missing_cols = [c for c in required if c not in raw_df.columns.str.strip()]
if missing_cols:
    st.error(f"The dataset is missing required columns: {', '.join(missing_cols)}")
    st.stop()

clean, model_df, quality_log = clean_data(raw_df)

tab_overview, tab_quality, tab_analysis, tab_model, tab_req, tab_diagrams = st.tabs([
    "Overview", "Data Quality", "Analysis", "Model", "Requirements", "Diagrams",
])

# ------------------------------- Overview ----------------------------------
with tab_overview:
    st.success("Dataset loaded successfully!")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{raw_df.shape[0]:,}")
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Eligible completed requests", f"{len(model_df):,}")

    st.subheader("First 10 rows")
    st.dataframe(raw_df.head(10), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Data types")
        st.dataframe(raw_df.dtypes.astype(str).to_frame("dtype"), use_container_width=True)
    with right:
        st.subheader("Missing values")
        st.dataframe(raw_df.isnull().sum().to_frame("Missing_Count"), use_container_width=True)

    st.subheader("Category frequencies")
    available = [c for c in CATEGORICAL_PREVIEW if c in raw_df.columns]
    selected = st.selectbox("Choose a column", available)
    st.dataframe(raw_df[selected].value_counts(dropna=False).to_frame("Count"),
                 use_container_width=True)

# ----------------------------- Data Quality --------------------------------
with tab_quality:
    st.subheader("Cleaning log")
    st.code("\n".join(quality_log), language="text")

    left, right = st.columns(2)
    with left:
        st.subheader("Remaining missing values")
        st.dataframe(clean.isnull().sum().to_frame("Missing_Count"), use_container_width=True)
    with right:
        st.subheader("Checks")
        st.metric("Duplicate IDs remaining", int(clean["Request_ID"].duplicated().sum()))
        st.metric("Cleaned dataset shape", f"{clean.shape[0]} × {clean.shape[1]}")

    st.subheader("Cleaned data (with engineered features)")
    st.dataframe(clean.head(10), use_container_width=True)

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Cleaned dataset", clean.to_csv(index=False).encode("utf-8"),
                       "student_service_requests_cleaned.csv", "text/csv")
    d2.download_button("⬇ Modeling subset", model_df.to_csv(index=False).encode("utf-8"),
                       "student_service_requests_modeling.csv", "text/csv")
    d3.download_button("⬇ Quality log", "\n".join(quality_log).encode("utf-8"),
                       "data_quality_log.txt", "text/plain")

# ------------------------------- Analysis ----------------------------------
with tab_analysis:
    if model_df.empty:
        st.warning("No eligible completed requests to analyze.")
    else:
        service_counts = model_df["Service_Type"].value_counts()
        service_breach = breach_table(model_df, "Service_Type").sort_values(
            "Breach_Rate", ascending=False
        )
        queue_breach = breach_table(model_df, "Queue_Category")

        st.subheader("Which service type has the most requests?")
        left, right = st.columns([1, 2])
        left.dataframe(service_counts.to_frame("Number_of_Requests"), use_container_width=True)
        with right:
            fig, ax = plt.subplots(figsize=(8, 5))
            service_counts.sort_values().plot(kind="barh", color="steelblue", ax=ax)
            ax.set_title("Completed Student Service Requests by Service Type")
            ax.set_xlabel("Number of Completed Requests")
            ax.set_ylabel("Service Type")
            fig.tight_layout()
            st.pyplot(fig)

        st.subheader("SLA breach rate by service type")
        st.dataframe(service_breach, use_container_width=True)

        st.subheader("SLA breach rate by queue category")
        left, right = st.columns([1, 2])
        left.dataframe(queue_breach, use_container_width=True)
        with right:
            queue_rates = (
                model_df.groupby("Queue_Category", observed=True)["SLA_Breached"]
                .mean().reindex(["Low", "Medium", "High"]).dropna() * 100
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            queue_rates.plot(kind="bar", color="darkorange", ax=ax)
            ax.set_title("SLA Breach Rate by Queue Category")
            ax.set_xlabel("Queue Category")
            ax.set_ylabel("SLA Breach Rate (%)")
            ax.tick_params(axis="x", rotation=0)
            fig.tight_layout()
            st.pyplot(fig)

        highest_service = service_breach.index[0]
        highest_rate = service_breach.iloc[0]["Breach_Rate_Percent"]
        st.info(f"**Highest observed service breach rate:** {highest_service} ({highest_rate}%)")

        if not queue_breach.empty:
            hq = queue_breach["Breach_Rate"].idxmax()
            hq_rate = queue_breach.loc[hq, "Breach_Rate"] * 100
            st.info(f"**Queue category with highest observed breach rate:** {hq} ({hq_rate:.2f}%)")

# -------------------------------- Model ------------------------------------
with tab_model:
    if model_df.empty:
        st.warning("No eligible completed requests to model.")
    else:
        result = train_model(model_df, max_depth, min_leaf, test_size, int(seed))

        if result is None:
            st.error(
                "The target contains only one class. "
                "A binary classification model requires both 0 and 1."
            )
        else:
            st.subheader("Target distribution")
            y = result["y"]
            c1, c2 = st.columns(2)
            c1.dataframe(y.value_counts().sort_index().to_frame("Count"), use_container_width=True)
            c2.dataframe((y.value_counts(normalize=True).sort_index() * 100).round(2)
                         .to_frame("Percent"), use_container_width=True)
            st.caption(f"Model dataset size: {len(y)} | "
                       f"Training records: {result['n_train']} | "
                       f"Testing records: {result['n_test']}")

            st.subheader("Decision Tree evaluation")
            cols = st.columns(4)
            for col, (name, val) in zip(cols, result["metrics"].items()):
                col.metric(name, f"{val:.3f}")

            left, right = st.columns(2)
            with left:
                st.markdown("**Classification report**")
                st.dataframe(result["report"].round(3), use_container_width=True)
            with right:
                st.markdown("**Confusion matrix**")
                fig, ax = plt.subplots(figsize=(5, 4))
                ConfusionMatrixDisplay(
                    confusion_matrix=result["cm"],
                    display_labels=["No Breach (0)", "Breach (1)"],
                ).plot(cmap="Blues", values_format="d", ax=ax)
                ax.set_title("Decision Tree Confusion Matrix")
                fig.tight_layout()
                st.pyplot(fig)

            st.subheader("Model comparison")
            comparison = pd.DataFrame({
                "Decision Tree": result["metrics"],
                "Majority Baseline": result["baseline_metrics"],
            })
            st.dataframe(comparison.round(3), use_container_width=True)

            st.subheader("Feature importance")
            imp = result["importance_df"].copy()
            imp["Original_Feature"] = imp["Encoded_Feature"].apply(original_feature)
            grouped = (
                imp.groupby("Original_Feature")["Importance"]
                .sum().sort_values(ascending=False)
            )

            left, right = st.columns(2)
            with left:
                st.markdown("**Top 10 encoded features**")
                st.dataframe(imp.head(10), use_container_width=True, hide_index=True)
                st.markdown("**Top three influential predictors**")
                st.dataframe(grouped.head(3).to_frame("Importance"), use_container_width=True)
            with right:
                fig, ax = plt.subplots(figsize=(7, 5))
                grouped.sort_values().plot(kind="barh", ax=ax)
                ax.set_title("Decision Tree Feature Importance")
                ax.set_xlabel("Importance")
                ax.set_ylabel("Predictor")
                fig.tight_layout()
                st.pyplot(fig)

# ----------------------------- Requirements --------------------------------
with tab_req:
    if model_df.empty:
        st.warning("No eligible data to derive requirements from.")
    else:
        volume = model_df["Service_Type"].value_counts()
        top_service, top_count = volume.index[0], int(volume.iloc[0])

        s_rates = model_df.groupby("Service_Type", observed=True)["SLA_Breached"].mean().dropna()
        q_rates = model_df.groupby("Queue_Category", observed=True)["SLA_Breached"].mean().dropna()

        hb_service = s_rates.idxmax() if not s_rates.empty else "No eligible data"
        hb_service_rate = s_rates.max() * 100 if not s_rates.empty else 0
        hb_queue = q_rates.idxmax() if not q_rates.empty else "No eligible data"
        hb_queue_rate = q_rates.max() * 100 if not q_rates.empty else 0

        st.markdown("#### Requirement 1 — Service Volume Monitoring")
        st.write(
            f"The system shall display request volume by service type. "
            f"Evidence: **{top_service}** has the highest completed request count "
            f"({top_count}) in this dataset."
        )
        st.markdown("#### Requirement 2 — SLA Breach Monitoring")
        st.write(
            f"The system shall calculate and display SLA breach rates by service type. "
            f"Evidence: **{hb_service}** has the highest observed rate ({hb_service_rate:.2f}%)."
        )
        st.markdown("#### Requirement 3 — Queue Risk Review")
        st.write(
            f"The system shall display SLA breach rates by queue category to help "
            f"administrators review workload. Evidence: **{hb_queue}** has the highest "
            f"observed queue-category rate ({hb_queue_rate:.2f}%)."
        )

# ------------------------------- Diagrams ----------------------------------
with tab_diagrams:
    st.subheader("UML Use Case Diagram")
    uc_fig = make_use_case_diagram()
    st.pyplot(uc_fig)
    st.download_button("⬇ Download use case diagram", fig_to_png(uc_fig),
                       "uml_use_case_diagram.png", "image/png")

    st.subheader("Dashboard Wireframe")
    wf_fig = make_wireframe()
    st.pyplot(wf_fig)
    st.download_button("⬇ Download wireframe", fig_to_png(wf_fig),
                       "dashboard_wireframe.png", "image/png")
