import math
import pandas as pd
from typing import Tuple

from .penalties import lookup_penalty_percent
from .rules import PBS_BASELINE_EDGE

def build_house_edge_report(
    csv_path: str,
    decisions_df: pd.DataFrame,
    pbs_baseline_edge: float = PBS_BASELINE_EDGE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - bets_df: round-by-round betting history per player
      - deviations_df: all deviations with penalty% and messages
      - session_summary_df: per-player session edge summary
    """
    df = pd.read_csv(csv_path)

    # Build deviations table from decisions_df
    devs = []
    for _, d in decisions_df.iterrows():
        if d["verdict"] != "Deviates":
            continue
        cards = d["player_cards"].split()
        up = d["dealer_upcard"]
        taken = d["action_taken"]
        pbs = d["pbs_action"]
        pct = lookup_penalty_percent(cards, up, taken, pbs)  # None if unknown
        devs.append({
            "timestamp_utc": d["timestamp_utc"],
            "player_id": d["player_id"],
            "hand_index": d["hand_index"],
            "stage": d["stage"],
            "player_cards": d["player_cards"],
            "dealer_upcard": up,
            "action_taken": taken,
            "pbs_action": pbs,
            "penalty_pct": pct,
            "explanation": d["message"].replace("This deviation increases expected losses.", "").strip()
        })
    deviations_df = pd.DataFrame(devs)

    base = df[[
        "timestamp_utc","player_id","base_bet","result","bankroll_before","bankroll_after","dealer_upcard","player_hands_json"
    ]].copy()

    base["realized_edge_pct"] = (-base["result"] / base["base_bet"]).replace([math.inf, -math.inf], pd.NA)

    if not deviations_df.empty:
        agg_pen = deviations_df.groupby(["timestamp_utc","player_id"], as_index=False)["penalty_pct"].sum(min_count=1)
    else:
        agg_pen = pd.DataFrame(columns=["timestamp_utc","player_id","penalty_pct"])

    bets_df = base.merge(agg_pen, on=["timestamp_utc","player_id"], how="left")
    bets_df["penalty_pct"] = bets_df["penalty_pct"].fillna(0.0)
    bets_df["expected_edge_pbs_pct"] = pbs_baseline_edge * 100.0
    bets_df["expected_edge_actual_pct"] = (pbs_baseline_edge + bets_df["penalty_pct"]) * 100.0

    def _mk_sentence(row):
        if row["penalty_pct"] == 0.0:
            return None
        inc = row["penalty_pct"] * 100.0
        return (f"Player {row['player_id']} bet ${row['base_bet']:.2f}. "
                f"Deviations this round increased expected house edge by {inc:.2f}% "
                f"(baseline {pbs_baseline_edge*100:.2f}% → {row['expected_edge_actual_pct']:.2f}%).")
    bets_df["round_penalty_sentence"] = bets_df.apply(_mk_sentence, axis=1)

    def _weighted_avg(group, col_pct):
        w = group["base_bet"].astype(float)
        x = group[col_pct].astype(float)
        if w.sum() == 0:
            return float("nan")
        return (w * x).sum() / w.sum()

    sess = []
    for pid, g in bets_df.groupby("player_id"):
        total_wager = g["base_bet"].sum()
        total_result = g["result"].sum()
        realized_edge = (-total_result / total_wager) * 100.0 if total_wager > 0 else float("nan")
        expected_pbs = _weighted_avg(g, "expected_edge_pbs_pct")
        expected_actual = _weighted_avg(g, "expected_edge_actual_pct")
        penalty_total = (expected_actual - expected_pbs) if (not math.isnan(expected_actual) and not math.isnan(expected_pbs)) else float("nan")
        sess.append({
            "player_id": pid,
            "rounds": len(g),
            "total_wager": total_wager,
            "total_result": total_result,
            "realized_house_edge_pct": realized_edge,
            "expected_house_edge_pbs_pct": expected_pbs,
            "expected_house_edge_actual_pct": expected_actual,
            "expected_penalty_addition_pct": penalty_total
        })
    session_summary_df = pd.DataFrame(sess).sort_values("player_id").reset_index(drop=True)

    if not deviations_df.empty:
        bb = base[["timestamp_utc","player_id","base_bet"]]
        deviations_df = deviations_df.merge(bb, on=["timestamp_utc","player_id"], how="left")
        deviations_df["penalty_dollars"] = deviations_df["penalty_pct"] * deviations_df["base_bet"]
        def _dev_sentence(r):
            inc = (r["penalty_pct"] or 0.0) * 100.0
            return (f"Player {r['player_id']} bet ${r['base_bet']:.2f}. {r['explanation']} "
                    f"This mistake increased the expected house edge by {inc:.2f}% for the round.")
        deviations_df["sentence"] = deviations_df.apply(_dev_sentence, axis=1)

    return bets_df, deviations_df, session_summary_df
