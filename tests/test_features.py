from t20predict.features import parse_match


def sample_match():
    return {"info": {"match_type": "T20", "overs": 20, "balls_per_over": 6,
                     "dates": ["2020-01-01"], "outcome": {"winner": "B"}},
            "innings": [{"team": "A", "overs": [{"over": 0, "deliveries": [
                {"runs": {"total": 2}}]}]}, {"team": "B", "overs": [
                    {"over": 0, "deliveries": [
                        {"runs": {"total": 1}, "extras": {"wides": 1}},
                        {"runs": {"total": 2}, "wickets": []}]}]}]}


def test_wide_does_not_use_a_legal_ball():
    rows, reason = parse_match(sample_match(), "example")
    assert reason is None
    assert rows[0]["balls_remaining"] == 120
    assert rows[0]["runs_needed"] == 2
    assert rows[1]["balls_remaining"] == 119
    assert rows[1]["runs_needed"] == 0


def test_dls_excluded():
    match = sample_match()
    match["info"]["outcome"]["method"] = "D/L"
    rows, reason = parse_match(match, "example")
    assert rows == [] and reason == "unsupported_match"


def test_normal_recorded_target_is_allowed_but_revised_target_is_not():
    match = sample_match()
    match["innings"][1]["target"] = {"overs": 20, "runs": 3}
    assert parse_match(match, "example")[1] is None
    match["innings"][1]["target"]["runs"] = 2
    assert parse_match(match, "example")[1] == "revised_target"


def test_result_not_in_features():
    from t20predict.features import FEATURES
    assert "chase_won" not in FEATURES
    assert "match_id" not in FEATURES


def test_final_score_changes_do_not_affect_prior_state():
    match = sample_match()
    rows, _ = parse_match(match, "example")
    assert rows[0]["runs_needed"] == 2
