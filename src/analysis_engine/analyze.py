import json
import pandas as pd
from typing import List, Tuple

from .rules import Rules
from .cards import is_pair, pair_rank, hand_totals_from_cards
from .pbs import pbs_recommendation
from .decisions import DecisionCheck, describe, classify_deviation, style_from_counts

def analyze_csv_against_pbs(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - decisions_df: one row per analyzed decision
      - players_summary_df: style classification per player_id
    """
    df = pd.read_csv(csv_path)

    def parse_rules(meta_str: str) -> Rules:
        try:
            meta = json.loads(meta_str)
        except Exception:
            return Rules()
        text = str(meta.get("rule_set", "")).lower()
        h17 = ("hits_soft_17=true" in text) or ("h17" in text)
        das = "double_after_split=true" in text
        surrender = "surrender_allowed=true" in text
        return Rules(
            decks=meta.get("num_decks", 6),
            h17=True if not isinstance(h17, bool) else h17,
            das=True if not isinstance(das, bool) else das,
            surrender=True if not isinstance(surrender, bool) else surrender,
        )

    decisions: List[DecisionCheck] = []

    for _, row in df.iterrows():
        timestamp = row.get("timestamp_utc", "")
        player_id = int(row.get("player_id"))
        dealer_up = str(row.get("dealer_upcard"))
        try:
            hands = json.loads(row["player_hands_json"])
        except Exception:
            hands = row["player_hands_json"]
        try:
            actions = json.loads(row["player_actions_json"])
        except Exception:
            actions = []
        rules = parse_rules(row.get("meta_json", "{}"))

        # Split decision (if any)
        if "split" in [str(a).lower() for a in actions]:
            if len(hands) >= 2:
                fr1 = hands[0]["cards"][0]
                fr2 = hands[1]["cards"][0]
                if fr1[:-1] == fr2[:-1]:
                    pair_cards = [hands[0]["cards"][0], hands[1]["cards"][0]]
                    pbs = pbs_recommendation(pair_cards, dealer_up, rules, True, False, True)
                    taken = "split"
                    verdict = "Complies" if taken == pbs else "Deviates"
                    note = f"Split decision on pair of {pair_cards[0][:-1]}{pair_cards[0][:-1]} vs dealer {dealer_up}: PBS recommends {pbs}."
                    decisions.append(DecisionCheck(
                        timestamp, player_id, -1, "split-decision", pair_cards, dealer_up, taken, pbs, verdict, note
                    ))

        # Per-hand evaluation
        for h_idx, h in enumerate(hands):
            cards = h["cards"]
            tot, soft = hand_totals_from_cards(cards)
            doubled = bool(h.get("doubled", False))
            surrendered = bool(h.get("surrendered", False))
            busted = bool(h.get("busted", False))

            first_two = cards[:2] if len(cards) >= 2 else cards
            first_decision = True
            can_double = (len(first_two) == 2)
            can_split = (len(first_two) == 2 and is_pair(first_two))

            if surrendered:
                taken_first = "surrender"
            elif doubled:
                taken_first = "double"
            elif len(cards) == 2:
                taken_first = "stand"
            else:
                taken_first = "hit"

            pbs_first = pbs_recommendation(first_two, dealer_up, rules, first_decision, can_double, can_split)
            verdict_first = "Complies" if taken_first == pbs_first else "Deviates"
            human_first = describe(first_two, dealer_up, taken_first, pbs_first)

            decisions.append(DecisionCheck(
                timestamp, player_id, h_idx, "initial", first_two, dealer_up, taken_first, pbs_first, verdict_first, human_first
            ))

            if taken_first == "hit" and not surrendered and not doubled:
                taken_final = "hit" if busted else "stand"
                pbs_final = pbs_recommendation(cards, dealer_up, rules, False, False, False)
                verdict_final = "Complies" if taken_final == pbs_final else "Deviates"
                human_final = describe(cards, dealer_up, taken_final, pbs_final, final=True)
                decisions.append(DecisionCheck(
                    timestamp, player_id, h_idx, "final", cards, dealer_up, taken_final, pbs_final, verdict_final, human_final
                ))

    # DataFrames
    dec_rows = []
    for d in decisions:
        dec_rows.append({
            "timestamp_utc": d.timestamp,
            "player_id": d.player_id,
            "hand_index": d.hand_index,
            "stage": d.stage,
            "player_cards": " ".join(d.player_cards),
            "dealer_upcard": d.dealer_upcard,
            "action_taken": d.action_taken,
            "pbs_action": d.pbs_action,
            "verdict": d.verdict,
            "message": d.note,
            "style_component": classify_deviation(d.action_taken, d.pbs_action),
        })
    decisions_df = pd.DataFrame(dec_rows).sort_values(["timestamp_utc", "player_id", "hand_index", "stage"])

    # Style summary
    style_summ = []
    for pid, grp in decisions_df.groupby("player_id"):
        cons = (grp["style_component"] == "Conservative").sum()
        aggr = (grp["style_component"] == "Aggressive").sum()
        devs = (grp["verdict"] == "Deviates").sum()
        total = len(grp)
        style = style_from_counts(cons, aggr, devs)
        style_summ.append({
            "player_id": pid,
            "decisions": total,
            "deviations": int(devs),
            "conservative_deviations": int(cons),
            "aggressive_deviations": int(aggr),
            "style": style
        })
    players_summary_df = pd.DataFrame(style_summ).sort_values("player_id")

    return decisions_df, players_summary_df
