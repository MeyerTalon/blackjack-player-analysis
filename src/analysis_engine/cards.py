from typing import List, Tuple

TEN_RANKS = {"T", "J", "Q", "K", "0"}  # include "0" only if '10' is normalized to 'T'

def rank_str(card: str) -> str:
    """
    Return rank as 'A', '2'..'9', 'T' for any 10-value (10/J/Q/K).
    Input cards look like 'QS', '10D', 'AC'.
    """
    s = card.upper()
    rank = s[:-1] if s[-1].isalpha() and len(s) > 1 else s
    rank = rank.replace("10", "T")
    if rank in {"J", "Q", "K"}:
        rank = "T"
    return rank

def dealer_upcard_value(up: str) -> int:
    r = rank_str(up)
    if r == "A":
        return 11
    if r == "T":
        return 10
    return int(r)

def hand_totals_from_cards(cards: List[str]) -> Tuple[int, bool]:
    """
    Compute best total and soft flag from list of cards.
    Ace counts as 11 where possible.
    Returns (total, is_soft).
    """
    values = []
    aces = 0
    for c in cards:
        r = rank_str(c)
        if r == "A":
            aces += 1
            values.append(11)
        elif r == "T":
            values.append(10)
        else:
            values.append(int(r))
    total = sum(values)
    soft = False
    if aces > 0:
        soft = True
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        soft = (aces > 0)
    return total, soft

def is_pair(cards2: List[str]) -> bool:
    return len(cards2) == 2 and rank_str(cards2[0]) == rank_str(cards2[1])

def pair_rank(cards2: List[str]) -> str:
    return rank_str(cards2[0])
