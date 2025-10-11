"""Game-tree construction and EV analysis for blackjack (infinite-deck model).

This module builds a recursive decision/chance tree for a player hand versus a
dealer upcard under configurable rules. It provides:
- `Rules`: table parameters (H17/S17, DAS, surrender, blackjack payout, resplits)
- Rank/hand utilities and dealer-final distribution under an infinite deck
- EV calculators (stand/double) and a full tree builder (`build_tree`)
- Simple viewers (`print_tree`, `tree_to_json`) for inspection/debugging
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

# -----------------------------
# Parameters / rule set
# -----------------------------
@dataclass(frozen=True)
class Rules:
    """
    Blackjack table parameters used by the tree/EV calculations.

    Attributes:
        h17: If True, dealer hits soft 17 (H17); otherwise stands (S17).
        das: If True, doubling after splits is allowed.
        surrender: Whether surrender is considered in EV calculations.
        blackjack_pays: Blackjack payout ratio (num, den), e.g., (3, 2) for 3:2.
        max_resplits: Maximum number of resplits permitted (3 ⇒ up to 4 hands).
    """
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
    """
    Normalizes a card string to a rank in {A,2..9,T}.

    Accepts inputs like '10H', 'QS', '7D', 'AC', or just '10', 'Q', '7', 'A'.

    Args:
        card: A rank or rank+suited card string (case-insensitive).

    Returns:
        str: One of 'A','2',...,'9','T'.
    """
    s = card.upper()
    r = s[:-1] if len(s) > 1 and s[-1].isalpha() else s
    r = r.replace("10","T")
    if r in {"J","Q","K"}: r = "T"
    return r

def card_value_rank(r: str) -> int:
    """Maps a normalized rank to its blackjack value (A=11, T=10, else numeric)."""
    if r == "A": return 11
    if r == "T": return 10
    return int(r)

# ---------------------------------------
# Hand totals / state helpers (player)
# ---------------------------------------
def hand_totals_from_ranks(ranks: List[str]) -> Tuple[int, bool]:
    """
    Computes best total and softness for a hand described by ranks.

    The algorithm starts with all Aces as 11 and demotes as needed to avoid busts.

    Args:
        ranks: List of normalized ranks (A,2..9,T).

    Returns:
        Tuple[int, bool]: (best_total, is_soft) where is_soft indicates whether an
        Ace is counted as 11 in the best total without busting.
    """
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
    """True if exactly two cards total 21 (natural blackjack)."""
    if len(ranks) != 2: return False
    total, _ = hand_totals_from_ranks(ranks)
    return total == 21

def is_pair(ranks: List[str]) -> bool:
    """True if the two-card hand consists of equal ranks (e.g., '8','8')."""
    return len(ranks) == 2 and ranks[0] == ranks[1]

# ---------------------------------------
# Dealer distribution (infinite deck)
# ---------------------------------------
# We compute the distribution over dealer final results given the upcard.
# Outcomes: keys are 17,18,19,20,21,'bust'. Values are probabilities.
# Infinite-deck i.i.d. recursion with memoization.

from functools import lru_cache

def _dealer_step_dist(total: int, soft: bool, rules: Rules) -> Dict[Any, float]:
    """
    Recursively computes dealer outcome distribution from (total, soft) state.

    Args:
        total: Current dealer total.
        soft: Whether the total is soft (some Ace counted as 11).
        rules: Dealer hit/stand configuration.

    Returns:
        Dict[Any, float]: Distribution over {17,18,19,20,21,'bust'} from this state.
    """
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
    """
    Computes dealer final distribution given an upcard (infinite deck).

    Args:
        up_rank: Dealer upcard rank normalized to {A,2..9,T}.
        rules: Dealer hit/stand configuration.

    Returns:
        Dict[Any, float]: Probability distribution over {17,18,19,20,21,'bust'}.
    """
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
    """
    Expected value (per 1 unit stake) if the player stands on `player_total`.

    Args:
        player_total: Player’s standing total (hard or soft, already evaluated).
        dealer_dist: Dealer terminal distribution.

    Returns:
        float: EV in units of the original wager (win=+1, loss=-1, push=0).
    """
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
    """
    Internal builder state carried during expansion.

    Attributes:
        rules: Table rules in effect.
        dealer_up: Dealer upcard rank (A,2..9,T).
        allow_split: Whether a split is legal at the current node.
        split_depth: Number of splits already performed.
        origin_is_split: True if this hand originated from a split.
    """
    rules: Rules
    dealer_up: str                 # rank "A","2"..,"9","T"
    allow_split: bool              # can split this two-card pair now
    split_depth: int               # how many splits taken so far
    origin_is_split: bool          # current hand originates from a split (affects e.g., blackjack counting)

def _terminal_node(ev: float, info: Dict[str,Any], prob: Optional[float]) -> Dict[str, Any]:
    """Creates a terminal node dictionary."""
    return {
        "node": "terminal",
        "info": info,
        "prob": prob,
        "ev": ev,
        "children": []
    }

def _decision_node(info: Dict[str,Any], prob: Optional[float]) -> Dict[str, Any]:
    """Creates a decision node dictionary with no children yet."""
    return {
        "node": "decision",
        "info": info,
        "prob": prob,
        "ev": 0.0,
        "children": []
    }

def _chance_node(info: Dict[str,Any], prob: Optional[float]) -> Dict[str, Any]:
    """Creates a chance node dictionary with no children yet."""
    return {
        "node": "chance",
        "info": info,
        "prob": prob,
        "ev": 0.0,
        "children": []
    }

def available_actions(player_ranks: List[str], st: BuildState) -> List[str]:
    """
    Returns legal actions for the current player hand and state.

    Actions considered: stand, hit, double (two-card only; DAS constrained), split (pair and resplit limits).

    Args:
        player_ranks: Player hand ranks (normalized).
        st: Current build state.

    Returns:
        List[str]: Subset of ["stand","hit","double","split"].
    """
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
    """
    Returns EV for a natural blackjack (if applicable), else None.

    Natural blackjacks from split hands are not counted as blackjack here.

    Args:
        player_ranks: Player ranks (two-card start).
        st: Current build state.
        dealer_dist: Dealer terminal distribution.

    Returns:
        Optional[float]: EV for a natural blackjack (push vs dealer 21, win at payout),
        or None if not applicable.
    """
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

def build_tree(
        player_cards: List[str],
        dealer_upcard: str,
        rules: Optional[Rules] = None,
        previous_actions: Optional[List[str]] = None
) -> Dict[str,Any]:
    """
    Builds a decision/chance tree for the given player hand vs dealer upcard.

    Args:
        player_cards: Player cards (e.g., ["9H","7D"]; ranks-only also supported).
        dealer_upcard: Dealer upcard (suit optional).
        rules: Optional table rules; defaults to `Rules()`.
        previous_actions: Informational only in this minimal API.

    Returns:
        Dict[str, Any]: Root node of the constructed game tree.
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

def _expand_decision(
        pranks: List[str],
        st: BuildState,
        prob: Optional[float]
) -> Dict[str,Any]:
    """
    Expands a decision node for the current player ranks and state.

    This function handles terminal checks (bust, blackjack), enumerates legal
    actions, and recursively builds chance/terminal children, computing EVs.

    Args:
        pranks: Player hand ranks (normalized).
        st: Current build state.
        prob: Probability of reaching this node from the parent (None at root).

    Returns:
        Dict[str, Any]: Decision/chance/terminal node with EV and children.
    """
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
    """
    Serializes a tree to pretty-printed JSON.

    Args:
        tree: Root node of the decision/chance tree.

    Returns:
        str: JSON string with indentation.
    """
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(x) for x in obj]
        return obj
    return json.dumps(clean(tree), indent=2)

def print_tree(
        node: Dict[str, Any],
        indent: str = "",
        prob_path: float = 1.0,
        show_probs=True
) -> None:
    """
    Pretty-prints the decision/chance/terminal tree to stdout.

    Args:
        node: Current node to print.
        indent: Indentation prefix for nested nodes.
        prob_path: Path probability multiplier used for display.
        show_probs: If True, prints branch probabilities at terminals.
    """
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
    print(tree_to_json(tree))
