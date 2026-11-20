import pandas as pd

from t20predict.model import split_chronologically


def test_split_keeps_dates_and_matches_disjoint():
    frame = pd.DataFrame({"match_id": [f"m{i}" for i in range(20)],
                          "date": [f"2020-01-{i+1:02d}" for i in range(20)]})
    train, val, test = split_chronologically(frame)
    assert train.date.max() < val.date.min() < val.date.max() < test.date.min()
    assert set(train.match_id).isdisjoint(val.match_id)
    assert set(val.match_id).isdisjoint(test.match_id)
