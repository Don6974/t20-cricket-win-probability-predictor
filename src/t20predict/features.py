"""Cricsheet JSON -> post-delivery chase states."""
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

FEATURES = ["runs_needed", "balls_remaining", "wickets_remaining", "target", "current_run_rate", "required_run_rate"]


def parse_match(match, match_id):
    info = match.get("info", {})
    innings = match.get("innings", [])
    outcome = info.get("outcome", {})
    if (info.get("match_type") != "T20" or info.get("balls_per_over", 6) != 6
            or info.get("overs", 20) != 20 or len(innings) != 2
            or "winner" not in outcome or "method" in outcome
            or any("super_over" in inn or "penalty_target" in inn for inn in innings)
            or any("eliminator" in inn for inn in innings)):
        return [], "unsupported_match"
    first, chase = innings
    if first.get("team") == chase.get("team") or any(i.get("overs") is None for i in innings):
        return [], "unsupported_match"
    # Exclude shortened innings, where 120-ball assumptions would be wrong.
    if any("powerplays" in i and any(p.get("type") == "mandatory" and p.get("to", 0) < 5 for p in i.get("powerplays", [])) for i in innings):
        return [], "unsupported_match"
    target = 1 + sum(d["runs"]["total"] for over in first["overs"] for d in over["deliveries"])
    # Cricsheet routinely includes the ordinary (not rain-revised) target.
    recorded_target = chase.get("target", {})
    if recorded_target and (recorded_target.get("runs") != target or recorded_target.get("overs", 20) != 20):
        return [], "revised_target"
    # Target is known before the second innings; no final-score or future features.
    chasing_team = chase["team"]
    rows, runs, legal, wickets = [], 0, 0, 0
    for over in chase["overs"]:
        for delivery in over["deliveries"]:
            runs += delivery["runs"]["total"]
            extras = delivery.get("extras", {})
            if not ("wides" in extras or "noballs" in extras):
                legal += 1
            wickets += sum(w.get("kind") not in {"retired hurt", "obstructing the field", "retired out"} for w in delivery.get("wickets", []))
            if legal > 120 or wickets > 10:
                return [], "invalid_state"
            remaining = 120 - legal
            needed = max(0, target - runs)
            rows.append({"match_id": match_id, "date": str(info["dates"][0]),
                         "gender": info.get("gender", "unknown"), "chasing_team": chasing_team,
                         "runs_needed": needed, "balls_remaining": remaining,
                         "wickets_remaining": 10 - wickets, "target": target,
                         "current_run_rate": 6 * runs / max(legal, 1),
                         "required_run_rate": 6 * needed / max(remaining, 1),
                         "chase_won": int(outcome["winner"] == chasing_team)})
            if needed == 0 or remaining == 0 or wickets >= 10:
                break
        if rows and (rows[-1]["runs_needed"] == 0 or rows[-1]["balls_remaining"] == 0 or rows[-1]["wickets_remaining"] == 0):
            break
    if not rows or (rows[-1]["runs_needed"] == 0) != (outcome["winner"] == chasing_team):
        # Avoid anomalous/forfeit/incorrectly labelled results.
        return [], "inconsistent_result"
    return rows, None


def load_archive(archive_path):
    records, skipped = [], Counter()
    with ZipFile(archive_path) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith(".json"))
        for name in names:
            try:
                match = json.loads(archive.read(name))
                rows, reason = parse_match(match, Path(name).stem)
                if reason:
                    skipped[reason] += 1
                else:
                    records.extend(rows)
            except (KeyError, TypeError, ValueError):
                skipped["parse_error"] += 1
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No eligible matches in archive")
    return frame.sort_values(["date", "match_id", "balls_remaining"], ascending=[True, True, False]).reset_index(drop=True), skipped
