"""Train, serialize, and evaluate a chronological match-disjoint model."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .features import FEATURES, load_archive


def split_chronologically(frame, train_fraction=0.7, validation_fraction=0.15):
    dates = frame.groupby("match_id")["date"].first().sort_values()
    unique_dates = sorted(dates.unique())
    if len(unique_dates) < 10:
        raise ValueError("Need >=10 distinct match dates")
    cut1 = unique_dates[int(len(unique_dates) * train_fraction)]
    cut2 = unique_dates[int(len(unique_dates) * (train_fraction + validation_fraction))]
    return (frame[frame.date < cut1].copy(),
            frame[(frame.date >= cut1) & (frame.date < cut2)].copy(),
            frame[frame.date >= cut2].copy())


def scored_rows(model, frame):
    probability = model.predict_proba(frame[FEATURES])[:, 1]
    # Terminal chase states are known with certainty and should not require ML.
    probability = np.where(frame.runs_needed == 0, 1.0, probability)
    probability = np.where((frame.balls_remaining == 0) | (frame.wickets_remaining == 0),
                           np.where(frame.runs_needed > 0, 0.0, probability), probability)
    return np.asarray(probability)


def metrics(frame, probabilities):
    truth = frame.chase_won.to_numpy()
    bins = np.linspace(0, 1, 11)
    calibration = []
    for i in range(10):
        mask = (probabilities >= bins[i]) & (probabilities < bins[i + 1] if i < 9 else probabilities <= bins[i + 1])
        if mask.any():
            calibration.append({"predicted_mean": round(float(probabilities[mask].mean()), 4),
                                "observed_win_rate": round(float(truth[mask].mean()), 4),
                                "n": int(mask.sum())})
    return {"accuracy": float(accuracy_score(truth, probabilities >= 0.5)),
            "log_loss": float(log_loss(truth, np.clip(probabilities, 1e-12, 1 - 1e-12), labels=[0, 1])),
            "brier_score": float(brier_score_loss(truth, probabilities)),
            "calibration_bins": calibration, "states": int(len(frame)),
            "matches": int(frame.match_id.nunique()),
            "date_min": str(frame.date.min()), "date_max": str(frame.date.max())}


def train(archive, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame, skipped = load_archive(archive)
    training, validation, test = split_chronologically(frame)
    model = HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=15,
                                           learning_rate=0.06, min_samples_leaf=100,
                                           l2_regularization=3.0, random_state=42)
    # State rows are correlated within a match. Equalize each match's total training weight.
    counts = training.groupby("match_id").match_id.transform("size")
    model.fit(training[FEATURES], training.chase_won, sample_weight=1 / counts)
    results = {"source": "Cricsheet all T20 JSON archive", "feature_names": FEATURES,
               "skipped_matches": dict(skipped), "train": metrics(training, scored_rows(model, training)),
               "validation": metrics(validation, scored_rows(model, validation)),
               "test": metrics(test, scored_rows(model, test))}
    joblib.dump(model, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
