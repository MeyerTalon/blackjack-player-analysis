"""
Utilities for serializing blackjack round data and constructing player objects.

This module provides helpers to serialize hands/players into JSON-ready dicts,
append per-round results to a CSV log, and build player instances for simulation.
"""

import csv, os, json
from datetime import datetime
from typing import Any, Dict, List
from src.blackjack_simulator.player import LLMBlackjackPlayer, RandomPlayer

def _format_card(card: Any) -> str:
    """
    Formats a card object into a compact string.

    Attempts attributes in order (rank/suit, val/suit) and falls back to `str(card)`.

    Args:
        card: Object representing a card; may expose `rank`, `suit`, or `val`.

    Returns:
        str: Compact card string such as "AH", "10D", or `str(card)` if unknown.
    """
    if hasattr(card, "rank") and hasattr(card, "suit"):
        return f"{card.rank}{getattr(card, 'suit', '')}"
    if hasattr(card, "val") and hasattr(card, "suit"):
        return f"{card.val}{card.suit}"
    return str(card)

def _serialize_hand(hand: Any) -> Dict[str, Any]:
    """
    Serializes a hand into a JSON-ready dictionary.

    The serializer is resilient to partially implemented hand interfaces and
    conditionally includes totals, flags (blackjack, busted, etc.), and wager.

    Args:
        hand: Hand-like object with optional attributes/methods such as
            `cards`, `best_total()`, `is_blackjack()`, `is_busted()`,
            `is_surrendered`, `is_doubled`, `is_split_aces`,
            `originated_from_split`, and `wager`.

    Returns:
        Dict[str, Any]: Dictionary encoding cards, totals, flags, and wager.
    """
    return {
        "cards": [_format_card(c) for c in getattr(hand, "cards", [])],
        "total": hand.best_total() if hasattr(hand, "best_total") else None,
        "blackjack": bool(getattr(hand, "is_blackjack", lambda: False)()),
        "busted": bool(getattr(hand, "is_busted", lambda: False)()),
        "surrendered": bool(getattr(hand, "is_surrendered", False)),
        "doubled": bool(getattr(hand, "is_doubled", False)),
        "split_aces": bool(getattr(hand, "is_split_aces", False)),
        "originated_from_split": bool(getattr(hand, "originated_from_split", False)),
        "wager": getattr(hand, "wager", None),
    }

def serialize_player_block(pstat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and normalizes per-player round statistics.

    Args:
        pstat: Raw player statistics for a round (from simulator).

    Returns:
        Dict[str, Any]: Dictionary with `result`, `player_totals`, serialized
        `player_hands`, and `player_actions`.
    """
    hands = pstat.get("player_hands", [])
    return {
        "result": pstat.get("result"),
        "player_totals": pstat.get("player_totals"),
        "player_hands": [_serialize_hand(h) for h in hands],
        "player_actions": pstat.get("player_actions", []),
    }

def append_round_to_csv(
    table_stats: Dict[str, Any],
    csv_path: str,
    round_meta: Dict[str, Any] = None,
    players_base_bets: List[float] = None,
    bankrolls_before: List[float] = None,
    bankrolls_after: List[float] = None,
) -> None:
    """
    Append round results to CSV. Accepts either single-player or multi-player stats.

    Args:
        table_stats: The 'stats' dict returned by play_round(...) or play_round_multi(...).
        csv_path: Where to write/append CSV.
        round_meta: Extra metadata (e.g., rule name, shoe penetration) merged into each row.
        players_base_bets: Optional list of base bets per player for this round.
        bankrolls_before: Optional list of pre-round bankrolls (per player).
        bankrolls_after: Optional list of post-round bankrolls (per player).
    Returns:
        None

    Raises:
        OSError: If the CSV cannot be created or written.
        ValueError: If `table_stats` is malformed for serialization.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # Dealer block
    dealer_hand = table_stats.get("dealer_hand")
    dealer_ser = _serialize_hand(dealer_hand) if dealer_hand is not None else None
    dealer_total = table_stats.get("dealer_total")
    dealer_up = table_stats.get("dealer_upcard")
    dealer_up_str = _format_card(dealer_up) if dealer_up is not None else None

    # Determine if this is multi-player or single-player
    # Multi-player has a "players" list; single-player has per-hand fields directly.
    if "players" in table_stats and isinstance(table_stats["players"], list):
        player_blocks = table_stats["players"]
    else:
        # Build a pseudo multi-player array of length 1
        player_blocks = [{
            "result": table_stats.get("result"),
            "player_totals": table_stats.get("player_totals"),
            "player_hands": table_stats.get("player_hands"),
            "player_actions": table_stats.get("player_actions", []),
        }]

    # CSV header
    fieldnames = [
        "timestamp_utc",
        "player_id",
        "table_id",
        "result",
        "base_bet",
        "bankroll_before",
        "bankroll_after",
        "dealer_total",
        "dealer_upcard",
        "dealer_hand_json",
        "player_totals",
        "player_hands_json",
        "player_actions_json",
        "meta_json",
    ]

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    with open(csv_path, mode="a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for i, p in enumerate(player_blocks):
            p_ser = serialize_player_block(p)
            row = {
                "timestamp_utc": now_iso,
                "player_id": i,
                "table_id": 1,  # this is a placeholder, there is only 1 table
                "result": p_ser["result"],
                "base_bet": (players_base_bets[i] if players_base_bets and i < len(players_base_bets) else None),
                "bankroll_before": (bankrolls_before[i] if bankrolls_before and i < len(bankrolls_before) else None),
                "bankroll_after": (bankrolls_after[i] if bankrolls_after and i < len(bankrolls_after) else None),
                "dealer_total": dealer_total,
                "dealer_upcard": dealer_up_str,
                "dealer_hand_json": json.dumps(dealer_ser, ensure_ascii=False),
                "player_totals": json.dumps(p_ser["player_totals"], ensure_ascii=False),
                "player_hands_json": json.dumps(p_ser["player_hands"], ensure_ascii=False),
                "player_actions_json": json.dumps(p_ser["player_actions"], ensure_ascii=False),
                "meta_json": json.dumps(round_meta or {}, ensure_ascii=False),
            }
            w.writerow(row)

def build_players(n_players: int, player_types: List[str], model: str, temperature: float):
    """
    Constructs a list of player instances for a table.

    If a single `player_types` entry is provided and `n_players > 1`, the type
    is replicated across seats. `"random"` yields `RandomPlayer`; any other
    string yields `LLMBlackjackPlayer` with the given persona.

    Args:
        n_players (int): Number of player seats to populate.
        player_types (List[str]): Types/personas per seat (e.g., ["basic","random"]).
        model (str): Model name for LLM players.
        temperature (float): Temperature for LLM players.

    Returns:
        List[Any]: List of player instances (`RandomPlayer` or `LLMBlackjackPlayer`).

    Raises:
        ValueError: If no types are provided or the counts do not match `n_players`.
    """
    if len(player_types) == 0:
        raise ValueError("At least one --player-type must be provided.")

    if len(player_types) == 1 and n_players > 1:
        player_types = player_types * n_players

    if len(player_types) != n_players:
        raise ValueError(f"--n-players ({n_players}) must match number of --player-type entries ({len(player_types)}).")

    players = []
    for t in player_types:
        if t.lower() == "random":
            players.append(RandomPlayer())
        else:
            players.append(LLMBlackjackPlayer(persona=t, model=model, temperature=temperature))
    return players


def build_base_bets(n_players: int, base_bet_value: float = None, base_bets_list: List[float] = None) -> List[float]:
    """
    Creates a per-player base-bet vector.

    Provide exactly one of `base_bet_value` (replicated) or `base_bets_list`
    (explicit per-seat values).

    Args:
        n_players (int): Number of player seats.
        base_bet_value (Optional[float]): Single base bet replicated across players.
        base_bets_list (Optional[List[float]]): Explicit per-player base bets.

    Returns:
        List[float]: Base bet for each player seat.

    Raises:
        ValueError: If both or neither sources are provided, or list length mismatches `n_players`.
    """
    if base_bets_list and base_bet_value is not None:
        raise ValueError("Provide either --base-bet or --base-bets, not both.")

    if base_bets_list:
        if len(base_bets_list) != n_players:
            raise ValueError(f"--base-bets length ({len(base_bets_list)}) must equal --n-players ({n_players}).")
        return base_bets_list

    if base_bet_value is None:
        raise ValueError("You must provide --base-bet (single) or --base-bets (list).")

    return [base_bet_value] * n_players