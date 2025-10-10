from typing import List
from .rules import Rules
from .cards import dealer_upcard_value, hand_totals_from_cards, is_pair, pair_rank

def pbs_recommendation(
    cards: List[str],
    dealer_up: str,
    rules: Rules,
    first_decision: bool,
    can_double: bool,
    can_split: bool,
) -> str:
    """
    Multi-deck, H17, DAS, late surrender PBS.
    Returns one of: 'hit', 'stand', 'double', 'split', 'surrender'.
    """
    up = dealer_upcard_value(dealer_up)
    tot, soft = hand_totals_from_cards(cards)

    # Late surrender (first decision, hard 2-card hands)
    if rules.surrender and first_decision and len(cards) == 2 and not soft:
        if tot == 16 and up in {9, 10, 11}:
            return "surrender"
        if tot == 15 and up == 10:
            return "surrender"

    # Pairs
    if first_decision and len(cards) == 2 and can_split and is_pair(cards):
        pr = pair_rank(cards)
        if pr == "A":  return "split"
        if pr == "T":  return "stand"
        if pr == "9":  return "split" if up in {2, 3, 4, 5, 6, 8, 9} else "stand"
        if pr == "8":  return "split"
        if pr == "7":  return "split" if 2 <= up <= 7 else "hit"
        if pr == "6":  return "split" if 2 <= up <= 6 else "hit"
        if pr == "5":
            if can_double and 2 <= up <= 9: return "double"
            return "hit"
        if pr == "4":
            return "split" if rules.das and up in {5, 6} else "hit"
        if pr in {"3", "2"}:
            return "split" if rules.das and 2 <= up <= 7 else "hit"

    # Soft totals
    if soft:
        if tot >= 19:
            if tot == 19 and first_decision and len(cards) == 2 and can_double and up == 6:
                return "double"
            return "stand"
        if tot == 18:
            if first_decision and len(cards) == 2 and can_double and 3 <= up <= 6:
                return "double"
            if up in {2, 7, 8}: return "stand"
            return "hit"
        if tot == 17:
            if first_decision and len(cards) == 2 and can_double and 3 <= up <= 6:
                return "double"
            return "hit"
        if tot in {16, 15}:
            if first_decision and len(cards) == 2 and can_double and 4 <= up <= 6:
                return "double"
            return "hit"
        if tot in {14, 13}:
            if first_decision and len(cards) == 2 and can_double and 5 <= up <= 6:
                return "double"
            return "hit"

    # Hard totals
    if tot >= 17: return "stand"
    if tot == 16: return "stand" if 2 <= up <= 6 else "hit"
    if tot == 15: return "stand" if 2 <= up <= 6 else "hit"
    if tot in {13, 14}: return "stand" if 2 <= up <= 6 else "hit"
    if tot == 12: return "stand" if 4 <= up <= 6 else "hit"
    if tot == 11:
        if first_decision and len(cards) == 2 and can_double: return "double"
        return "hit"
    if tot == 10:
        if first_decision and len(cards) == 2 and can_double and 2 <= up <= 9: return "double"
        return "hit"
    if tot == 9:
        if first_decision and len(cards) == 2 and can_double and 3 <= up <= 6: return "double"
        return "hit"
    return "hit"  # 8 or less
