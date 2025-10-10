from dataclasses import dataclass
from typing import List, Tuple, Any

from .cards import hand_totals_from_cards, is_pair, pair_rank

@dataclass
class DecisionCheck:
    timestamp: str
    player_id: int
    hand_index: int
    stage: str               # 'initial' | 'final' | 'split-decision'
    player_cards: List[str]
    dealer_upcard: str
    action_taken: str
    pbs_action: str
    verdict: str             # 'Complies' | 'Deviates'
    note: str

def classify_deviation(action_taken: str, pbs_action: str) -> str:
    """Conservative | Aggressive | Neutral (for complies)."""
    if action_taken == pbs_action:
        return "Neutral"
    conservative_patterns = [
        (action_taken == "stand" and pbs_action == "hit"),
        (action_taken in {"hit", "stand"} and pbs_action in {"double", "split"}),
        (action_taken == "surrender" and pbs_action in {"hit", "stand", "double"}),
    ]
    if any(conservative_patterns):
        return "Conservative"
    aggressive_patterns = [
        (action_taken == "hit" and pbs_action == "stand"),
        (action_taken in {"double", "split"} and pbs_action in {"hit", "stand"}),
    ]
    if any(aggressive_patterns):
        return "Aggressive"
    return "Neutral"

def style_from_counts(cons: int, aggr: int, total_dev: int) -> str:
    if total_dev == 0:
        return "Neutral"
    if cons > aggr:
        return "Conservative"
    if aggr > cons:
        return "Aggressive"
    return "Neutral"

def cards_pretty(cards: List[str]) -> str:
    return "[" + ", ".join(cards) + "]"

def describe(cards: List[str], dealer_up: str, taken: str, pbs: str, final: bool=False) -> str:
    total, soft = hand_totals_from_cards(cards)
    kind = "soft" if soft else "hard"
    stage = "final" if final else "initial"
    if len(cards) == 2 and is_pair(cards):
        desc = f"Pair of {pair_rank(cards)}s vs dealer {dealer_up}"
    else:
        desc = f"{kind.capitalize()} {total} vs dealer {dealer_up}"
    if taken == pbs:
        return f"{desc}. Player {taken} ({stage}). PBS agrees."
    else:
        harm = "This deviation increases expected losses."
        return f"{desc}. Player {taken} ({stage}). PBS recommends {pbs}. {harm}"
