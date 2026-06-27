"""Credit-Risk Analyst Dashboard (Streamlit).

A human-facing surface over the deployed model: pick a customer, see their
calibrated probability of default, the risk band, and the SHAP reason codes
behind the score — and run "what-if" scenarios on the latest statement. This is
the on-demand, single-account view a risk analyst would use (the portfolio is
scored in batch; see serving/app/batch_score.py). It calls the *same*
CreditModel and feature pipeline as the API, so the numbers match production.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO.parent))            # serving/  -> import app.*
from app.model import CreditModel               # noqa: E402
from app.pipeline import DATE_COL, ID_COL        # noqa: E402

BAND_COLOR = {"very low": "#1a9850", "low": "#91cf60", "medium": "#fee08b",
              "high": "#fc8d59", "very high": "#d73027"}


@st.cache_resource
def get_model() -> CreditModel:
    return CreditModel()


@st.cache_data
def get_samples():
    # JSON (not parquet) so the small preset bundle ships inside the image —
    # *.parquet is git-ignored and would be stripped from the build context.
    df = pd.read_json(DEMO / "sample_customers.json")
    meta = json.loads((DEMO / "sample_meta.json").read_text())
    return df, meta


st.set_page_config(page_title="AMEX Credit-Risk Analyst", page_icon="💳",
                   layout="wide")
model, (samples, meta) = get_model(), get_samples()

st.title("💳 Credit-Risk Analyst Dashboard")
st.caption(
    "Look up a customer's **probability of default (PD)** and the reasons behind "
    "it. Model: calibrated LightGBM (AUC 0.96, KS 0.79) scoring 13 months of "
    "anonymized statements. The portfolio is scored in **batch**; this is the "
    "on-demand single-account view.")

# --- sidebar: choose a customer + what-if sliders ---------------------------
st.sidebar.header("Customer")
labels = [m["label"] for m in meta]
choice = st.sidebar.selectbox("Profile (real test customers)", labels, index=2)
cid = next(m["customer_ID"] for m in meta if m["label"] == choice)
cust = samples[samples[ID_COL] == cid].sort_values(DATE_COL).reset_index(drop=True)
last = cust.iloc[-1]

st.sidebar.header("What-if: latest statement")
st.sidebar.caption("Adjust the most recent month and watch the PD and reasons "
                   "respond.")


def _slider(col, label, lo, hi):
    cur = float(last[col]) if pd.notna(last[col]) else 0.0
    return st.sidebar.slider(label, lo, hi, max(lo, min(hi, cur)))


mod = cust.copy()
mod.loc[mod.index[-1], "P_2"] = _slider("P_2", "Payment level (P_2)", -1.0, 2.0)
mod.loc[mod.index[-1], "D_39"] = _slider("D_39", "Delinquency (D_39)", 0.0, 5.0)
mod.loc[mod.index[-1], "B_1"] = _slider("B_1", "Balance (B_1)", 0.0, 2.0)

result = model.score(mod)[0]
pd_pct = result["probability_of_default"] * 100
band = result["risk_band"]

# --- top row: gauge + band + reason codes -----------------------------------
left, right = st.columns([1, 1.3])
with left:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=pd_pct,
        number={"suffix": "%"},
        title={"text": "Probability of default"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": BAND_COLOR.get(band, "#555")},
               "steps": [{"range": [0, 20], "color": "#e8f5e9"},
                         {"range": [20, 50], "color": "#fff8e1"},
                         {"range": [50, 100], "color": "#ffebee"}]}))
    gauge.update_layout(height=300, margin=dict(t=40, b=0))
    st.plotly_chart(gauge, use_container_width=True)
    st.markdown(f"### Risk band: "
                f"<span style='color:{BAND_COLOR.get(band)}'>{band.upper()}</span>",
                unsafe_allow_html=True)

with right:
    st.subheader("Why — top risk drivers (SHAP)")
    reasons = result["top_reason_codes"]
    if reasons:
        rdf = pd.DataFrame(reasons).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=rdf["contribution"], y=rdf["description"], orientation="h",
            marker_color="#d73027"))
        fig.update_layout(height=300, margin=dict(t=10, b=10, l=10),
                          xaxis_title="contribution to risk (log-odds)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No features are pushing this customer toward default — low risk.")

# --- bottom: the customer's payment trajectory ------------------------------
st.subheader("Payment trajectory (P_2 over the last 13 statements)")
traj = cust[[DATE_COL, "P_2"]].copy()
traj[DATE_COL] = pd.to_datetime(traj[DATE_COL]).dt.strftime("%Y-%m")
st.line_chart(traj.set_index(DATE_COL))

with st.expander("About this model / disclaimer"):
    st.markdown(
        "- **Model:** calibrated LightGBM, the production model of record "
        "(a LightGBM+XGBoost+GRU blend scored marginally higher but isn't worth "
        "the production complexity / explainability cost).\n"
        "- **Features:** 1,628 per-customer aggregates of 13 monthly statements; "
        "the same code computes them for training, the API and this app (no skew).\n"
        "- **Not a lending decision tool.** A demonstration on anonymized public "
        "Kaggle data; real deployment would add fairness testing, auth and a "
        "model registry.")
