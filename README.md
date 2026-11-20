# 🏏 T20 Cricket Live Win Probability Predictor

**Author:** Sugandhika Dontha · Cricket Analyst

[Try the T20 Live Win Probability Predictor](https://dontha-t20-predictor.streamlit.app/)

Enter the target, current runs, completed overs,
balls and wickets to estimate the chasing team's
win probability.

[View Source Code](https://github.com/Don6974/t20-cricket-win-probability-predictor)

An end-to-end Python portfolio project estimating the probability that the **team batting second** wins a standard 20-over T20 match from its **current post-delivery state**. Includes a reproducible Cricsheet data pipeline, chronological out-of-sample evaluation, unit tests and an interactive Streamlit demonstration.

> **Not a betting profitability model.** No historical bookmaker odds, market timing, commissions, limits, slippage or staking returns are evaluated. These probabilities should not be interpreted as profitable wagers.

## Data and reproducibility

- Source: [Cricsheet T20 JSON download](https://cricsheet.org/downloads/t20s_json.zip), [format documentation](https://cricsheet.org/format/json/), [licensing and attribution](https://cricsheet.org/matches/). Cricsheet JSON data is released under the Open Data Commons Attribution License.
- The archive is **downloaded locally** and omitted from Git to avoid distributing a large changing dataset. Dataset updates will change the metrics; record the downloaded archive checksum when rerunning.
- Dataset snapshot used for the original run (19 November 2026), SHA-256: `956f29df9d2b95dd6fe12de5bc70d4522f23693e5e7e2ac4c1a004e1b541ad8e`.
- Use Python 3.10+.

```bash
git clone https://github.com/Don6974/t20-cricket-win-probability-predictor.git
cd t20-cricket-win-probability-predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
mkdir -p data/raw
curl -L https://cricsheet.org/downloads/t20s_json.zip -o data/raw/t20s_json.zip
python -m pytest -q
python train.py
streamlit run app.py
```

On Windows without `curl`, manually download the ZIP and place it at `data/raw/t20s_json.zip`.

The original trained `artifacts/model.joblib` is included in this repository so the demo can start without downloading the training data. Retraining overwrites it. Only load `joblib` models from trusted sources.

## What the model sees

Each row is the state **immediately after a delivery**, counting wides and no-balls as deliveries without incrementing legal balls. Features:

| Input | Meaning |
| --- | --- |
| Runs needed | Target minus current chase runs, floored at zero |
| Balls remaining | 120 minus legal deliveries consumed |
| Wickets remaining | Ten minus counted dismissals |
| Target | First-innings total + 1, known before chase |
| Current run rate | Runs scored per six legal deliveries to date |
| Required run rate | Runs required per six remaining legal deliveries |

No final score, match result, winning team, next-ball events or information from future deliveries enters `FEATURES`. `chase_won` is used as the **training label only**. A completed successful chase returns 100%; an exhausted-ball or all-out unsuccessful chase returns 0%.

## Model and evaluation

- `HistGradientBoostingClassifier` with fixed hyperparameters and random seed.
- 70% earliest **match dates** for training; next 15% for validation; latest 15% for the untouched chronological test. All games on a date belong to the same partition. We do **not** random-split balls from a match.
- Training weights normalize each match's aggregate influence so unusually long innings do not dominate solely due to extra ball states.
- Accuracy uses a 0.5 cutoff. Log loss and Brier score measure probabilistic quality; ten equal-width bins provide calibration evidence.
- Each ball state shares its match's outcome label, so ball rows are correlated and row-wise metrics **are not independent-sample confidence intervals**. For robust scientific comparisons, add match-level uncertainty intervals, tournament/year and men's/women's subgroups, and benchmark models.
- Different domestic and international competitions are pooled. Team strength, individual batters, venues, pitch, and pre-match priors are **not included**; probabilities may be poorly calibrated in specific competitions or extreme scenarios.
- Excludes non-T20 and non-20-over records, ties/undecided matches, super-overs, recorded revised targets, recorded method outcomes (including D/L), inconsistent finishes, and unusable records. Missing metadata or incompletely recorded rain adjustments remain potential data-quality limitations.

Actual measured metrics (from a specific dataset snapshot) live in [`artifacts/metrics.json`](artifacts/metrics.json). The `test` section is the primary held-out result, and `calibration_bins` contain mean predicted probability, empirical win rate and observation count. No numbers are supplied without running the model.

### Results on the 19 November 2026 snapshot

| Chronological partition | Matches | In-play states | Accuracy | Log loss | Brier score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training (2005–Jun 2024) | 3,066 | 337,156 | 84.14% | 0.3331 | 0.1078 |
| Validation (Jun 2024–Sep 2025) | 1,064 | 116,333 | 84.94% | 0.3247 | 0.1036 |
| **Held-out test (Sep 2025–Nov 2026)** | **1,083** | **116,815** | **85.51%** | **0.3050** | **0.0981** |

**Calibration is imperfect.** On the test set, states with predictions averaging ~54.9% won only ~43.7%; predictions averaging ~75.1% won ~67.6%. This model is a first benchmark, not a reliably calibrated sportsbook price. Also, many very late-game states are close to certain, so full-innings accuracy can overstate the model's usefulness during genuinely uncertain periods.

## Repository layout

```text
app.py                        Streamlit match-state demo
train.py                      CLI for data parsing, training and metrics
src/t20predict/features.py    Cricsheet parser, in-play states and feature definitions
src/t20predict/model.py       Date splits, model fitting and scoring
tests/                        Parser and chronological split tests
artifacts/metrics.json        Measured evaluation and calibration bins
artifacts/model.joblib        Trained model (generated locally)
requirements*.txt             Runtime and test dependencies
```

## Deploy demo

1. Publish this repository, including the trained `artifacts/model.joblib`.
2. At [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app → Deploy a public app from GitHub**.
3. Set repository/branch and **main file: `app.py`**. Dependencies install from `requirements.txt`.
4. Open the resulting URL and verify that a chase like 90/3 after 10 overs targeting 175 returns a probability.

**Caveat:** This app predicts hypothetical cricket states. Do not use it for real-money trading without a separate, realistic, market-odds-based backtest and model risk controls.

## Portfolio roadmap

1. Compare to a naive rate/balls-remaining baseline and a logistic-regression baseline.
2. Add held-out match-level bootstrap intervals, time-stratified calibration and low-sample-size warnings.
3. Explore pre-match team strength that is computed *as of* the match date only, plus season-specific models.
4. Only with timestamped historical odds: evaluate value, closing-line, vig, transaction costs and realistic execution.

**Acknowledgement:** Thanks to [Cricsheet](https://cricsheet.org/) and its contributors for publishing structured ball-by-ball cricket data.
