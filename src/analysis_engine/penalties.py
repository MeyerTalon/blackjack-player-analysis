from typing import List, Tuple, Any, Optional
from .cards import rank_str as _rank_str

# Penalty table: (“hard”|“soft”|“pair”, total_or_pair, dealer_up_value, taken, pbs) -> penalty (fraction)
_PENALTY = {}

def _add_pen(kind, tot_or_rank, ups, taken, pbs, pct):
    for u in ups:
        _PENALTY[(kind, tot_or_rank, u, taken, pbs)] = pct

# Examples and magnitudes
_add_pen("hard", 12, [2], "stand", "hit", 0.0030)
_add_pen("hard", 12, [3], "stand", "hit", 0.0025)
_add_pen("hard", 16, [10, 11, 9], "stand", "hit", 0.0060)
_add_pen("hard", 15, [10], "stand", "surrender", 0.0030)
_add_pen("hard", 15, [10], "hit", "surrender", 0.0030)

for up in [10, 9, 11]:
    _add_pen("hard", 16, [up], "stand", "surrender", 0.0045)
    _add_pen("hard", 16, [up], "hit", "surrender", 0.0045)

_add_pen("hard", 11, [10, 11], "hit", "double", 0.0070)
_add_pen("hard", 11, [10, 11], "stand", "double", 0.0070)
_add_pen("hard", 10, [9], "hit", "double", 0.0060)
_add_pen("hard", 10, [9], "stand", "double", 0.0060)

for up in [3,4,5,6]:
    _add_pen("hard", 9, [up], "hit", "double", 0.0040)
    _add_pen("hard", 9, [up], "stand", "double", 0.0040)

_add_pen("soft", 18, [2,7,8], "hit", "stand", 0.0025)
for up in [3,4,5,6]:
    _add_pen("soft", 18, [up], "stand", "double", 0.0040)
    _add_pen("soft", 18, [up], "hit", "double", 0.0060)

for u in [2,3,4,5,6,7,8,9,10,11]:
    _add_pen("pair", "8", [u], "stand", "split", 0.0070)
    _add_pen("pair", "8", [u], "hit",   "split", 0.0070)

_add_pen("pair", "9", [2,3,4,5,6,8,9], "stand", "split", 0.0045)
_add_pen("pair", "9", [2,3,4,5,6,8,9], "hit",   "split", 0.0045)

_add_pen("pair", "4", [5,6], "hit", "split", 0.0030)
_add_pen("pair", "4", [5,6], "stand", "split", 0.0030)

for up in [2,3,4,5,6,7,8,9]:
    _add_pen("pair", "5", [up], "hit", "double", 0.0060)
    _add_pen("pair", "5", [up], "stand", "double", 0.0060)

def _dealer_up_value_str_to_val(up: str) -> int:
    up = up.upper()
    rank = up[:-1]
    if rank == "A":
        return 11
    if rank in {"10","J","Q","K"}:
        return 10
    return int(rank)

def _hand_kind_and_key(cards) -> tuple[str, str | Any, None] | tuple[str, int, None]:
    def rs(c):
        c = c.upper()
        rank = c[:-1]
        if rank in {"J","Q","K"}: rank = "10"
        return rank

    ranks = [rs(c) for c in cards]
    if len(cards) == 2 and ranks[0] == ranks[1]:
        return "pair", ranks[0], None

    values, aces = [], 0
    for r in ranks:
        if r == "A":
            values.append(11); aces += 1
        else:
            values.append(10 if r == "10" else int(r))
    total = sum(values)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    soft = (aces > 0) and (total <= 21)
    return "soft" if soft else "hard", total, None

def lookup_penalty_percent(player_cards, dealer_upcard, taken, pbs) -> float | None:
    kind, key, _ = _hand_kind_and_key(player_cards)
    upv = _dealer_up_value_str_to_val(dealer_upcard)
    return _PENALTY.get((kind, key, upv, taken, pbs))
