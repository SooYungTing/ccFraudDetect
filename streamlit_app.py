import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detector")
st.caption("Pre-trained model included. Test with built-in sample data or upload your own CSV.")

# Path
APP_DIR = pathlib.Path(__file__).parent
MODEL_PATH = APP_DIR / "fraud_model.pkl"  # model
SAMPLE_PATH = APP_DIR / "data" / "creditcard.csv" # demo CSV

#Load built-in model
@st.cache_resource
def load_model():
    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict):
        model = obj.get("model")
        features = obj.get("features")
    else:
        model = obj
        features = getattr(model, "feature_names_in_", None)
    return model, list(features) if features is not None else None

model, feature_cols = load_model()
if not hasattr(model, "predict_proba"):
    st.error("Loaded model does not support predict_proba().")
    st.stop()
st.success("Pre-trained model loaded automatically.")

# Sidebar: threshold 
threshold = st.sidebar.slider(
    "Decision threshold (fraud if proba >= T)",
    0.00, 1.00, 0.50, 0.01,
    help="Lower this to increase recall (more frauds flagged)."
)

#Helpers 
def align_features(df: pd.DataFrame, expected_cols: list[str] | None) -> pd.DataFrame:
    if expected_cols is None:
        return df
    missing = [c for c in expected_cols if c not in df.columns]
    extra   = [c for c in df.columns if c not in expected_cols]
    if missing:
        st.error(f"Missing required columns: {missing}")
    if extra:
        st.warning(f"Ignoring extra columns: {extra}")
    return df[[c for c in expected_cols if c in df.columns]]

def diagnostics(proba: np.ndarray, pred: np.ndarray, labels: np.ndarray | None):
    st.write(f"Decision threshold = **{threshold:.2f}**")
    st.write(f"Predicted positives: **{int(pred.sum())}** / {len(pred)} ({pred.mean():.2%})")
    st.write(f"fraud_proba → min **{proba.min():.6f}**, mean **{proba.mean():.6f}**, max **{proba.max():.6f}**")
    if proba.max() < 0.02:
        st.warning("All predicted probabilities are very low (<0.02). "
                   "This often happens if the sample has no frauds or the threshold is too high.")
    if labels is not None:
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        cm = confusion_matrix(labels, pred, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        prec, rec, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
        st.write("**Confusion matrix (rows=actual, cols=pred)**")
        st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))
        st.write(f"Precision **{prec:.3f}** • Recall **{rec:.3f}** • F1 **{f1:.3f}** "
                 f"• TP **{tp}**, FP **{fp}**, FN **{fn}**, TN **{tn}**")

def score(df: pd.DataFrame):
    X = df.copy()
    y = None
    if "Class" in X.columns:
        y = X["Class"].astype(int).to_numpy()
        X = X.drop(columns=["Class"])
    X = align_features(X, feature_cols)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    out = X.copy()
    out["fraud_proba"] = proba
    out["fraud_pred"]  = pred
    diagnostics(proba, pred, y)
    st.dataframe(out.head(25))
    st.download_button("⬇️ Download predictions",
                       out.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv",
                       mime="text/csv")

def one_row_df(values: dict[str, float], cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{c: values.get(c, 0.0) for c in cols}], columns=cols)

# how to stuffs
with st.expander("ℹ️ How to test", expanded=True):
    st.markdown("""
1. **Built-in sample:** click **"▶ Test with built-in sample data"** to see predictions immediately.
2. **Upload CSV:** provide a file with the same feature columns used in training (`Time, V1...V28, Amount`).
3. **Single transaction:** enter feature values manually and get a probability.

*Fraud is very rare (~0.17%). Lower the threshold (e.g. 0.10) or use a sample that contains some frauds to see positive predictions.*
""")

#  Tabs
tab_demo, tab_csv, tab_single = st.tabs([
    "▶ Built-in Sample Data",
    "📄 Upload CSV",
    "🧪 Single Transaction"
])

#  Built-in sample
with tab_demo:
    st.subheader("Built-in sample (Kaggle-style CSV)")
    if not SAMPLE_PATH.exists():
        st.warning("No sample CSV found at `data/sample_transactions.csv`. Add one to use this demo.")
    else:
        if st.button("▶ Test with built-in sample data"):
            df = pd.read_csv(SAMPLE_PATH)
            score(df)

#  Upload CSV
with tab_csv:
    st.subheader("Upload your own CSV of transactions")
    data_file = st.file_uploader("CSV with the same feature columns. "
                                 "If it contains 'Class', metrics will be shown.", type=["csv"], key="csv")
    if data_file:
        df = pd.read_csv(data_file)
        score(df)

# Single transaction
with tab_single:
    st.subheader("Enter a single transaction")
    if not feature_cols:
        st.info("Feature names aren't available from the model.")
    else:
        left, right = st.columns(2)
        inputs = {}
        half = (len(feature_cols) + 1) // 2
        for name in feature_cols[:half]:
            inputs[name] = left.number_input(name, value=0.0, step=0.1, format="%.6f")
        for name in feature_cols[half:]:
            inputs[name] = right.number_input(name, value=0.0, step=0.1, format="%.6f")
        if st.button("Predict single transaction"):
            X = one_row_df(inputs, feature_cols)
            proba = float(model.predict_proba(X)[:, 1][0])
            pred = int(proba >= threshold)
            st.metric("Fraud probability", f"{proba:.4f}")
            st.metric("Prediction (1 = fraud)", f"{pred}")
            st.caption(f"Decision threshold = {threshold:.2f}")
