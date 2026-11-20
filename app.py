from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

from sklearn._loss import _loss as sklearn_loss
from t20predict.features import FEATURES

# Compatibility fix for the serialized scikit-learn model.
sys.modules.setdefault("_loss", sklearn_loss)


st.set_page_config(page_title="T20 Chase Win Probability", page_icon="🏏", layout="centered")
st.title("🏏 T20 Live Chase Win Probability")
st.caption("Historical Cricsheet-based model · standard 20-over second-innings chases · not betting advice")
model_path = Path(__file__).parent / "artifacts" / "model.joblib"
if not model_path.exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

target = st.number_input("Target to win", min_value=1, max_value=350, value=175)
runs = st.number_input("Chasing team's runs", min_value=0, max_value=400, value=90)
overs = st.number_input("Completed overs", min_value=0, max_value=20, value=10)
balls = st.number_input("Legal balls in current over", min_value=0, max_value=5, value=0,
                        disabled=(overs == 20))
wickets = st.slider("Wickets lost", min_value=0, max_value=10, value=3)
legal = 6 * overs + (0 if overs == 20 else balls)
remaining = 120 - legal
needed = max(0, target - runs)
state = {"runs_needed": needed, "balls_remaining": remaining,
         "wickets_remaining": 10 - wickets, "target": target,
         "current_run_rate": 6 * runs / max(legal, 1),
         "required_run_rate": 6 * needed / max(remaining, 1)}
if needed == 0:
    p = 1.0
elif remaining == 0 or wickets == 10:
    p = 0.0
else:
    p = float(joblib.load(model_path).predict_proba(pd.DataFrame([state])[FEATURES])[0, 1])
st.metric("Chasing team win probability", f"{100 * p:.1f}%")
st.progress(float(p))
st.write(f"**{needed} runs needed · {remaining} legal balls remaining · {10-wickets} wickets in hand**")
st.info("This is a historical outcome probability, not a profitable betting signal. No bookmaker odds, margin or staking strategy is included.")
