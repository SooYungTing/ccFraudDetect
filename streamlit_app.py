import io, pathlib, re, tempfile
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detector")
st.caption("Model & sample CSV URLs are configured in secrets.toml")

# Helpers: load from Google Drive / GitHub blob/raw / HTTP / local
def _is_url(s: str) -> bool:
    return isinstance(s, str) and s.lower().startswith(("http://", "https://"))

def _to_raw_github(url: str) -> str:
    if "://github.com/" in url and "/blob/" in url:
        user_repo, rest = url.split("github.com/")[1].split("/blob/", 1)
        return f"https://raw.githubusercontent.com/{user_repo}/{rest}"
    return url

def _drive_file_id(url: str):
    m = re.search(r"/file/d/([^/]+)/", url)
    if m: return m.group(1)
    m = re.search(r"[?&]id=([^&]+)", url)
    if m: return m.group(1)
    return None

@st.cache_data(show_spinner=False)
def fetch_csv(spec: str) -> pd.DataFrame:
    """Load CSV from Google Drive, GitHub, plain URL or local path."""
    fid = _drive_file_id(spec)
    if fid:
        import gdown
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        gdown.download(id=fid, output=tmp.name, quiet=True)   # handles Drive confirm pages
        return pd.read_csv(tmp.name)

    spec = _to_raw_github(spec)
    if _is_url(spec):
        return pd.read_csv(spec)

    return pd.read_csv(pathlib.Path(spec))

@st.cache_resource
def fetch_model(spec: str):
    """Load joblib-saved model from Drive/GitHub/URL/local path."""
    fid = _drive_file_id(spec)
    if fid:
        import gdown
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        gdown.download(id=fid, output=tmp.name, quiet=True)
        obj = joblib.load(tmp.name)
    else:
        spec = _to_raw_github(spec)
        if _is_url(spec):
            resp = requests.get(spec, timeout=30)
            resp.raise_for_status()
            obj = joblib.load(io.BytesIO(resp.content))
        else:
            obj = joblib.load(pathlib.Path(spec))

    if isinstance(obj, dict):
        model = obj.get("model")
        features = obj.get("features")
    else:
        model = obj
        features = getattr(model, "feature_names_in_", None)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba().")

    return model, (list(features) if features is not None else None)

def align_features(df: pd.DataFrame, expected_cols: Optional[list]) -> pd.DataFrame:
    if not expected_cols:
        return df
    missing = [c for c in expected_cols if c not in df.columns]
    extra   = [c for c in df.columns if c not in expected_cols]
    if missing:
        st.error(f"Missing required columns: {missing}")
    if extra:
        st.warning(f"Ignoring extra columns: {extra}")
    return df[[c for c in expected_cols if c in df.columns]]

def diagnostics(proba: np.ndarray, pred: np.ndarray,
                y_true: Optional[np.ndarray], threshold: float):
    st.write(f"Decision threshold = **{threshold:.2f}**")
    st.write(f"Predicted positives: **{int(pred.sum())}** / {len(pred)} ({pred.mean():.2%})")
    st.write(f"fraud_proba → min **{proba.min():.6f}**, mean **{proba.mean():.6f}**, max **{proba.max():.6f}**")
    if proba.max() < 0.02:
        st.warning("All predicted probabilities are very low (<0.02). "
                   "This often happens if your sample has no frauds or the threshold is too high.")
    if y_true is not None:
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        cm = confusion_matrix(y_true, pred, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        st.write("**Confusion matrix (rows=actual, cols=pred)**")
        st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"],
                                  columns=["Pred 0","Pred 1"]))
        st.write(f"Precision **{prec:.3f}** • Recall **{rec:.3f}** • F1 **{f1:.3f}** • "
                 f"TP **{tp}**, FP **{fp}**, FN **{fn}**, TN **{tn}**")

def score(df: pd.DataFrame, model, feature_cols: Optional[list], threshold: float):
    X = df.copy()
    y = None
    if "Class" in X.columns:
        y = X["Class"].astype(int).to_numpy()
        X = X.drop(columns=["Class"])
    X = align_features(X, feature_cols)
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)
    out   = X.copy()
    out["fraud_proba"] = proba
    out["fraud_pred"]  = pred
    diagnostics(proba, pred, y, threshold)
    st.dataframe(out.head(25))
    st.download_button("⬇️ Download predictions",
                       out.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv",
                       mime="text/csv")

def one_row_df(values: dict, cols: list) -> pd.DataFrame:
    return pd.DataFrame([{c: values.get(c, 0.0) for c in cols}], columns=cols)

# Load model & sample CSV paths from secrets
MODEL_SPEC  = st.secrets["paths"]["model_path"]
SAMPLE_SPEC = st.secrets["paths"]["sample_csv"]

try:
    model, feature_cols = fetch_model(MODEL_SPEC)
    st.success("Model loaded from secrets.toml path.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Sidebar: threshold slider
threshold = st.sidebar.slider(
    "Decision threshold (fraud if proba ≥ T)",
    0.0, 1.0, 0.5, 0.01,
    help="Lower this to increase recall (flag more frauds)."
)

with st.expander("ℹ️ How to test", expanded=True):
    st.markdown("""
1. **Built-in sample** - click **"▶ Test with built-in sample data"** to see predictions immediately.  
2. **Upload CSV** - provide a file with the same feature columns used in training (`Time, V1...V28, Amount`).  
3. **Single transaction** - enter feature values manually.

*Fraud is very rare (~0.17%). Lower the threshold (e.g. 0.10) or use a sample that contains some frauds to see positive predictions.*
""")

tab_demo, tab_csv, tab_single = st.tabs([
    "▶ Built-in Sample Data",
    "📄 Upload CSV",
    "🧪 Single Transaction"
])

with tab_demo:
    st.subheader("Built-in sample (from secrets)")
    if st.button("▶ Test with built-in sample data"):
        try:
            df = fetch_csv(SAMPLE_SPEC)
            score(df, model, feature_cols, threshold)
        except Exception as e:
            st.error(f"Failed to load sample CSV: {e}")

with tab_csv:
    st.subheader("Upload your own CSV of transactions")
    data_file = st.file_uploader(
        "CSV with the same feature columns. "
        "If it contains 'Class', metrics will be shown.", type=["csv"], key="csv")
    if data_file:
        df = pd.read_csv(data_file)
        score(df, model, feature_cols, threshold)

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
            pred  = int(proba >= threshold)
            st.metric("Fraud probability", f"{proba:.4f}")
            st.metric("Prediction (1 = fraud)", f"{pred}")
            st.caption(f"Decision threshold = {threshold:.2f}")
