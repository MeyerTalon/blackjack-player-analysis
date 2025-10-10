# blackjack_tree.py
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

# -----------------------------
# Parameters / rule set
# -----------------------------
@dataclass(frozen=True)
class Rules:
    h17: bool = True                 # Dealer hits soft 17 (H17). If False, stands on soft 17 (S17)
    das: bool = True                 # Double after split allowed
    surrender: bool = True           # Used in EV here
    blackjack_pays: Tuple[int,int] = (3,2)   # 3:2 by default
    max_resplits: int = 3            # up to 3 resplits -> 4 hands

# Infinite-deck probabilities by rank (we treat any 10-value as T)
RANKS = ["A","2","3","4","5","6","7","8","9","T"]
RANK_PROB = {
    "A": 1/13,
    "2": 1/13, "3": 1/13, "4": 1/13, "5": 1/13, "6": 1/13,
    "7": 1/13, "8": 1/13, "9": 1/13,
    "T": 4/13,   # {10, J, Q, K}
}

def rank_of_card(card: str) -> str:
    """Normalize a card like '10H','QS','7D','AC' to ranks A,2..9,T"""
    s = card.upper()
    r = s[:-1] if len(s) > 1 and s[-1].isalpha() else s
    r = r.replace("10","T")
    if r in {"J","Q","K"}: r = "T"
    return r

def card_value_rank(r: str) -> int:
    if r == "A": return 11
    if r == "T": return 10
    return int(r)

# ---------------------------------------
# Hand totals / state helpers (player)
# ---------------------------------------
def hand_totals_from_ranks(ranks: List[str]) -> Tuple[int, bool]:
    """Return best total (<=21 if possible) and soft flag."""
    total = 0
    aces = 0
    for r in ranks:
        if r == "A":
            total += 11
            aces += 1
        elif r == "T":
            total += 10
        else:
            total += int(r)
    # reduce aces as needed
    soft = aces > 0
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    soft = (aces > 0) and (total <= 21)
    return total, soft

def is_blackjack(ranks: List[str]) -> bool:
    if len(ranks) != 2: return False
    total, _ = hand_totals_from_ranks(ranks)
    return total == 21

def is_pair(ranks: List[str]) -> bool:
    return len(ranks) == 2 and ranks[0] == ranks[1]

# ---------------------------------------
# Dealer distribution (infinite deck)
# ---------------------------------------
# We compute the distribution over dealer final results given the upcard.
# Outcomes: keys are 17,18,19,20,21,'bust'. Values are probabilities.
# Infinite-deck i.i.d. recursion with memoization.

from functools import lru_cache

def _dealer_step_dist(total: int, soft: bool, rules: Rules) -> Dict[Any, float]:
    """Recursively compute distribution from current dealer state (total/soft)."""
    # Terminal tests:
    if total > 21:
        return {"bust": 1.0}
    if total > 17:
        return {total: 1.0}  # 18..21
    if total == 17:
        if soft and rules.h17:
            pass  # must hit
        else:
            return {17: 1.0}
    # Must hit: draw a rank
    out: Dict[Any,float] = {}
    for r, p in RANK_PROB.items():
        v = 11 if r == "A" else (10 if r == "T" else int(r))
        t = total + v
        s = soft or (r == "A")
        # reduce aces if needed
        if s and t > 21:
            # Convert one ace value 11->1
            t -= 10
            # After conversion, determine if still soft (any other Ace 11 remains?)
            # For infinite-deck recursion, it's enough to recompute soft via logic below:
            # If an Ace counted as 11 still remains, we would have kept soft. Here we only
            # tracked "has some Ace 11?" But when we reduce exactly one, we may still have others.
            # Simpler: recompute soft by reverse-checking possibility of adding +10 <=21.
            # But we don't know ace count. We'll approximate: if t+10 <= 21 then soft True.
            s = (t + 10) <= 21
        sub = _dealer_step_dist(t, s, rules)
        for k, q in sub.items():
            out[k] = out.get(k, 0.0) + p*q
    return out

@lru_cache(None)
def dealer_distribution(up_rank: str, rules: Rules) -> Dict[Any, float]:
    """Dealer final distribution given an upcard rank (infinite deck)."""
    # Start by drawing the hole card, then resolve.
    out: Dict[Any,float] = {}
    up_val = 11 if up_rank == "A" else (10 if up_rank == "T" else int(up_rank))
    soft = (up_rank == "A")
    for r, p in RANK_PROB.items():
        v = 11 if r == "A" else (10 if r == "T" else int(r))
        total = up_val + v
        s = soft or (r == "A")
        if s and total > 21:
            total -= 10  # demote one Ace
            s = (total + 10) <= 21
        sub = _dealer_step_dist(total, s, rules)
        for k, q in sub.items():
            out[k] = out.get(k, 0.0) + p*q
    # Normalize tiny numeric drift
    s = sum(out.values())
    if s > 0:
        for k in list(out.keys()):
            out[k] /= s
    return out

# ---------------------------------------
# EV vs dealer (player stands/bust/double)
# ---------------------------------------
def ev_vs_dealer(player_total: int, dealer_dist: Dict[Any,float]) -> float:
    """EV (per 1 unit stake) when player stands on a given total."""
    ev = 0.0
    for k, p in dealer_dist.items():
        if k == "bust":
            ev += p * 1.0
        else:
            d = int(k)
            if player_total > d:
                ev += p * 1.0
            elif player_total < d:
                ev += p * -1.0
            else:
                ev += p * 0.0
    return ev

# ---------------------------------------
# Game tree building
# ---------------------------------------
# Node schema:
#   {
#     "node": "decision" | "chance" | "terminal",
#     "info": {...},            # readable state (hand, totals, etc.)
#     "prob": float,            # probability from parent (None for root)
#     "ev": float,              # EV from this node (per initial unit)
#     "children": [ ... ]       # list of nodes (if any)
#   }
#
# We measure EV in units of the original hand wager = 1.0.
# Double multiplies stake on that hand (so outcomes ×2).
# Split duplicates stake for a second hand (EV sums).

@dataclass
class BuildState:
    rules: Rules
    dealer_up: str                 # rank "A","2"..,"9","T"
    allow_split: bool              # can split this two-card pair now
    split_depth: int               # how many splits taken so far
    origin_is_split: bool          # current hand originates from a split (affects e.g., blackjack counting)

def _terminal_node(ev: float, info: Dict[str,Any], prob: Optional[float]) -> Dict[str,Any]:
    return {
        "node": "terminal",
        "info": info,
        "prob": prob,
        "ev": ev,
        "children": []
    }

def _decision_node(info: Dict[str,Any], prob: Optional[float]) -> Dict[str,Any]:
    return {
        "node": "decision",
        "info": info,
        "prob": prob,
        "ev": 0.0,
        "children": []
    }

def _chance_node(info: Dict[str,Any], prob: Optional[float]) -> Dict[str,Any]:
    return {
        "node": "chance",
        "info": info,
        "prob": prob,
        "ev": 0.0,
        "children": []
    }

def available_actions(player_ranks: List[str], st: BuildState) -> List[str]:
    total, soft = hand_totals_from_ranks(player_ranks)
    acts = []
    # Stand always legal
    acts.append("stand")
    # Hit always legal unless already busted (handled before decision)
    acts.append("hit")
    # Double allowed at exactly two cards; after split depends on DAS
    if len(player_ranks) == 2 and (st.rules.das or not st.origin_is_split):
        acts.append("double")
    # Split if exactly two cards, same rank, below max resplits
    if len(player_ranks) == 2 and is_pair(player_ranks) and st.allow_split and st.split_depth < st.rules.max_resplits:
        acts.append("split")
    return acts

def compute_blackjack_ev_if_applicable(player_ranks: List[str], st: BuildState, dealer_dist: Dict[Any,float]) -> Optional[float]:
    """If initial 2-card natural, return EV (3:2) except if from split (no natural). Else None."""
    if len(player_ranks) == 2 and is_blackjack(player_ranks) and not st.origin_is_split:
        # Natural BJ pushes against dealer BJ if that occurs in dist.
        # In our dealer dist we include outcomes after drawing hole card; includes 21 totals,
        # but not a "natural" flag. Approximate: treat any dealer total 21 as push (slight bias).
        # (Exact needs peek / composition; acceptable for this infinite-deck tree.)
        bj_num, bj_den = st.rules.blackjack_pays
        pay = bj_num / bj_den
        push_prob = dealer_dist.get(21, 0.0)  # crude approximation
        win_prob = 1.0 - push_prob
        return win_prob * pay  # pushes pay 0
    return None

def build_tree(player_cards: List[str],
               dealer_upcard: str,
               rules: Optional[Rules] = None,
               previous_actions: Optional[List[str]] = None) -> Dict[str,Any]:
    """
    Build game tree from current state.
    - player_cards: e.g. ["9H","7D"] or ranks ["9","7"] also accepted.
    - dealer_upcard: e.g. "6S" or "6" or "T" (we only use rank).
    - previous_actions: optional list like ["split"] (informative only in this minimal API).
    """
    rules = rules or Rules()
    pranks = [rank_of_card(c) for c in player_cards]
    up = rank_of_card(dealer_upcard)

    st = BuildState(
        rules=rules,
        dealer_up=up,
        allow_split=True,
        split_depth=0,
        origin_is_split=False
    )
    root = _expand_decision(pranks, st, prob=None)
    return root

def _expand_decision(pranks: List[str], st: BuildState, prob: Optional[float]) -> Dict[str,Any]:
    total, soft = hand_totals_from_ranks(pranks)
    dealer_dist = dealer_distribution(st.dealer_up, st.rules)

    # Bust check
    if total > 21:
        return _terminal_node(ev=-1.0, info={"state":"player_bust","hand":pranks,"total":total,"soft":soft}, prob=prob)

    # Natural BJ check (only at exactly two cards)
    if len(pranks) == 2:
        ev_bj = compute_blackjack_ev_if_applicable(pranks, st, dealer_dist)
        if ev_bj is not None:
            return _terminal_node(ev=ev_bj, info={"state":"player_blackjack","hand":pranks}, prob=prob)

    # Decision node
    node = _decision_node(info={"hand":pranks, "total":total, "soft":soft, "dealer_up":st.dealer_up}, prob=prob)
    acts = available_actions(pranks, st)

    # STAND
    if "stand" in acts:
        ev = ev_vs_dealer(total, dealer_dist)
        node["children"].append(_terminal_node(ev=ev, info={"action":"stand","total":total}, prob=1.0))

    # HIT
    if "hit" in acts:
        ch = _chance_node(info={"action":"hit_draw"}, prob=1.0)
        # draw each rank
        for r, p in RANK_PROB.items():
            new_hand = pranks + [r]
            ch["children"].append(_expand_decision(new_hand, st, prob=p))
        # chance EV = sum(p * child.ev)
        ch["ev"] = sum(c["prob"] * c["ev"] for c in ch["children"])
        node["children"].append(ch)

    # DOUBLE (draw one card, then stand; stake x2)
    if "double" in acts:
        ch = _chance_node(info={"action":"double_draw"}, prob=1.0)
        for r, p in RANK_PROB.items():
            new_hand = pranks + [r]
            tot2, _ = hand_totals_from_ranks(new_hand)
            if tot2 > 21:
                ev = -2.0
            else:
                ev = 2.0 * ev_vs_dealer(tot2, dealer_dist)
            ch["children"].append(_terminal_node(ev=ev, info={"after":"double","card":r,"result_total":tot2}, prob=p))
        ch["ev"] = sum(c["prob"] * c["ev"] for c in ch["children"])
        node["children"].append(ch)

    # SPLIT (if pair)
    if "split" in acts:
        # After split: two independent child hands, each gets one draw, then decisions continue.
        # EV = EV(hand L) + EV(hand R), stake duplicated (1 + 1).
        # Build a 2-card draw chance node for both children (13x13 outcomes).
        pair_rank = pranks[0]
        ch = _chance_node(info={"action":"split_deal_children","pair":pair_rank}, prob=1.0)

        # New split state
        st_child = BuildState(
            rules=st.rules,
            dealer_up=st.dealer_up,
            allow_split=True,  # allow re-splitting if ranks match again later
            split_depth=st.split_depth + 1,
            origin_is_split=True
        )

        for r1, p1 in RANK_PROB.items():
            left_hand = [pair_rank, r1]
            left_subtree = _expand_decision(left_hand, st_child, prob=None)  # prob applied at parent
            for r2, p2 in RANK_PROB.items():
                right_hand = [pair_rank, r2]
                right_subtree = _expand_decision(right_hand, st_child, prob=None)

                # combine EVs additively (two wagers)
                ev = left_subtree["ev"] + right_subtree["ev"]
                ch["children"].append({
                    "node":"terminal",
                    "info":{"split_children":[r1,r2]},
                    "prob": p1 * p2,
                    "ev": ev,
                    "children":[
                        {"role":"left", **left_subtree},
                        {"role":"right", **right_subtree},
                    ]
                })
        ch["ev"] = sum(c["prob"] * c["ev"] for c in ch["children"])
        node["children"].append(ch)

    # Node EV = max EV among decision children (optimal play from this state).
    if node["children"]:
        node["ev"] = max(child["ev"] if child["node"] != "chance" else child["ev"] for child in node["children"])
    else:
        node["ev"] = 0.0
    return node

# ---------------------------------------
# Tree viewers / JSON
# ---------------------------------------
def tree_to_json(tree: Dict[str,Any]) -> str:
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(x) for x in obj]
        return obj
    return json.dumps(clean(tree), indent=2)

def print_tree(node: Dict[str,Any], indent: str = "", prob_path: float = 1.0, show_probs=True):
    ntype = node["node"]
    ev = node.get("ev", 0.0)
    pstr = f" p={prob_path:.6f}" if show_probs and node.get("prob") is not None else ""
    if ntype == "decision":
        info = node.get("info", {})
        t = info.get("total")
        soft = info.get("soft")
        print(f"{indent}[DECISION] hand={info.get('hand')} total={t}{' (soft)' if soft else ''} EV*={ev:.4f}")
        for c in node["children"]:
            # for decisions, child.prob is the branch chance (1.0 for terminal stand)
            cp = c.get("prob", 1.0)
            print_tree(c, indent + "  ", prob_path*cp, show_probs)
    elif ntype == "chance":
        info = node.get("info", {})
        print(f"{indent}[CHANCE] {info}  EV={ev:.4f}")
        for c in node["children"]:
            cp = c.get("prob", 1.0)
            print_tree(c, indent + "  ", prob_path*cp, show_probs)
    else:  # terminal
        info = node.get("info", {})
        print(f"{indent}[TERMINAL]{pstr} info={info} EV={ev:.4f}")

# ---------------------------------------
# Example usage
# ---------------------------------------
if __name__ == "__main__":
    # Example: player 9,7 vs dealer 6 (common “16 vs 6”)
    tree = build_tree(["9H","7D"], "6S", Rules(h17=True, das=True))
    print("--- TEXT TREE (truncated suggestion: toggle below) ---")
    # Warning: hit/split trees can be large. For a quick peek, you might comment out printing chance sub-branches.
    print_tree(tree, show_probs=False)

    # JSON output (write if you want)
    # print(tree_to_json(tree))
