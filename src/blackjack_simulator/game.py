"""
Comprehensive blackjack simulation with support for LLM-controlled players.

This module implements:
- Configurable rules (`BlackjackRules`) and a multi-deck shoe (`Shoe`).
- Player hand mechanics (`Hand`) and dealer play logic.
- A lightweight view (`HandView`) for feeding decisions to policies/LLMs.
- Single-round (`play_round`) and multi-round (`play_many`) game loops.
- CLI entry point (`main`) to run simulations and log results to CSV.

It integrates with local Ollama models via `LLMBlackjackPlayer` and supports
baseline random play via `RandomPlayer`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal, Dict
import random
import argparse
from src.blackjack_simulator.player import LLMBlackjackPlayer, RandomPlayer
from src.blackjack_simulator.game_utils import append_round_to_csv
from src.blackjack_simulator.game_utils import build_players, build_base_bets
import ollama
from datetime import datetime

Action = Literal["hit", "stand", "double", "split", "surrender"]

# ---------------------------------------------------------------------------
# Rules and shoe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlackjackRules:
    """
    Table configuration used by the simulator and policy logic.

    Attributes:
        decks: Number of standard 52-card decks in the shoe.
        dealer_hits_soft_17: If True, dealer hits on soft 17 (H17); otherwise stands (S17).
        double_after_split: If True, allowing doubling after splits (DAS).
        surrender_allowed: If True, late surrender is available on the first decision.
        blackjack_pays_ratio: Payout ratio for a natural blackjack (numerator, denominator).
        resplit_limit: Maximum number of resplits allowed (e.g., 3 => up to 4 hands).
        resplit_aces_allowed: If True, aces may be resplit; otherwise restricted.
    """
    decks: int = 6
    dealer_hits_soft_17: bool = True  # H17=True, S17=False
    double_after_split: bool = True
    surrender_allowed: bool = True     # Late surrender
    blackjack_pays_ratio: Tuple[int, int] = (3, 2)  # 3:2
    resplit_limit: int = 3             # up to 3 resplits (4 hands)
    resplit_aces_allowed: bool = False

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["S", "H", "D", "C"]

def make_deck() -> List[str]:
    """
    Builds a single 52-card deck as rank-suit strings (e.g., 'AS', '10D').

    Returns:
        the deck list.
    """
    return [f"{r}{s}" for r in RANKS for s in SUITS]

class Shoe:
    """
    Multi-deck shoe with cut-card penetration and shuffle management.

    Cards are stored as rank-suit strings. The shoe tracks a cut index to signal
    when a reshuffle is needed based on the `penetration` fraction.

    Args:
        decks: Number of decks to include in the shoe.
        penetration: Fraction of shoe consumed before triggering reshuffle.
        rng: Optional random number generator for deterministic tests.

    Attributes:
        decks: Number of decks.
        penetration: Cut-card fraction.
        cards: Current stack of cards (top at the end).
        dealt: Number of cards dealt since the last shuffle.
        cut_index: Index after which `need_shuffle` becomes True.
        need_shuffle: Flag set when penetration reached.
    """
    def __init__(self, decks: int = 6, penetration: float = 0.75, rng: Optional[random.Random] = None):
        assert decks >= 1 and 0.0 < penetration < 1.0
        self.decks = decks
        self.penetration = penetration
        self.rng = rng or random.Random()
        self._cards_init = make_deck() * decks
        self.cards = list(self._cards_init)
        self.dealt = 0
        self.cut_index = int(len(self.cards) * self.penetration)
        self.shuffle()
        self.need_shuffle = False

    def shuffle(self) -> None:
        """Resets and shuffles the shoe; clears dealt count and `need_shuffle`."""
        self._cards_init = make_deck() * self.decks
        self.cards = list(self._cards_init)
        self.rng.shuffle(self.cards)
        self.dealt = 0

    def deck_size(self) -> int:
        """Returns the number of cards currently remaining in the shoe."""
        return len(self.cards)

    def draw(self, n: int = 1) -> List[str]:
        """
        Draws `n` cards from the top of the shoe. Sets need_shuffle as necessary.

        Args:
            n: Number of cards to draw.

        Returns:
            List[str]: Drawn cards (rank-suit strings).

        Raises:
            IndexError: If attempting to draw more cards than remain.
        """
        if n > self.deck_size():
            raise IndexError(f'Trying to draw {n} cards from deck of size {self.deck_size()}.')
        out = []
        for _ in range(n):
            out.append(self.cards.pop())
            self.dealt += 1
            if self.dealt >= self.cut_index:  # reached cut card
                self.need_shuffle = True
        return out

# ---------------------------------------------------------------------------
# Hand utilities
# ---------------------------------------------------------------------------

def card_value(rank: str) -> int:
    """Maps a rank to its blackjack value (A=11, 10/J/Q/K=10, else numeric)."""
    if rank in ("J", "Q", "K", "10"):
        return 10
    if rank == "A":
        return 11
    return int(rank)

def split_rank(card: str) -> str:
    """Returns the rank portion of a card string (everything but the final suit)."""
    # rank is everything except last char (suit)
    return card[:-1]

@dataclass
class Hand:
    """
    Mutable representation of a player's (or dealer's) hand and flags.

    Attributes:
        cards: List of card strings.
        wager: Current wager attached to this hand.
        is_finished: Whether this hand is done acting.
        is_doubled: Whether the hand has doubled down.
        is_surrendered: Whether the hand was surrendered.
        is_split_aces: Whether this hand resulted from splitting aces.
        originated_from_split: Whether the hand is from any split (aces or not).
    """
    cards: List[str]
    wager: float
    is_finished: bool = False
    is_doubled: bool = False
    is_surrendered: bool = False
    is_split_aces: bool = False       # track split aces rule (usually one-card only)
    originated_from_split: bool = False

    def add(self, card: str) -> None:
        """Adds a card to the hand."""
        self.cards.append(card)

    def ranks(self) -> List[str]:
        """Returns the rank strings of all cards in the hand."""
        return [split_rank(c) for c in self.cards]

    def totals(self) -> Tuple[int, Optional[int]]:
        """
        Computes (hard_total, soft_total_or_None).

        The soft total counts a single Ace as 11 when it does not cause a bust.

        Returns:
            Tuple[int, Optional[int]]: Hard total and optional soft total.
        """
        hard = sum(card_value(r) if r != "A" else 1 for r in self.ranks())
        aces = sum(1 for r in self.ranks() if r == "A")
        soft = None
        if aces:
            # try to upgrade one Ace to 11 if it doesn't bust
            if hard + 10 <= 21:
                soft = hard + 10
        return hard if soft is None else soft - 10, soft

    def best_total(self) -> int:
        """Returns the highest non-busting total (soft if valid, else hard)."""
        hard, soft = self.totals()
        return soft if soft is not None and soft <= 21 else hard

    def is_soft(self) -> bool:
        """True if the hand has a valid soft total (<= 21)."""
        _, soft = self.totals()
        return soft is not None and soft <= 21

    def is_blackjack(self) -> bool:
        """True if the hand is a natural blackjack (two cards totaling 21)."""
        return len(self.cards) == 2 and self.best_total() == 21

    def is_busted(self) -> bool:
        """True if the best total exceeds 21."""
        return self.best_total() > 21

    def is_pair(self) -> bool:
        """True if the hand has exactly two cards of the same rank."""
        return len(self.cards) == 2 and split_rank(self.cards[0]) == split_rank(self.cards[1])

    def can_split(self, rules: BlackjackRules, splits_done: int) -> bool:
        """Whether the hand can be split under the given rules and split count."""
        if not self.is_pair():
            return False
        if splits_done >= rules.resplit_limit:
            return False
        # handle resplitting aces restriction
        if split_rank(self.cards[0]) == "A" and not rules.resplit_aces_allowed and self.originated_from_split:
            return False
        return True

# ---------------------------------------------------------------------------
# Dealer logic
# ---------------------------------------------------------------------------

def dealer_initial_play(shoe: Shoe) -> Tuple[Tuple, bool]:
    """
    Deals dealer upcard/hole and peeks for blackjack when applicable.

    Args:
        shoe: Shared shoe to draw from.

    Returns:
        ((upcard, holecard), dealer_has_blackjack): Tuple of dealer cards plus
        a flag indicating whether peek detected a natural blackjack.
    """
    dealer_up = shoe.draw(1)[0]
    dealer_hole = shoe.draw(1)[0]
    dealer_has_blackjack = False

    # Peek for dealer blackjack if upcard is A or 10
    dealer_has_peek = split_rank(dealer_up) in ("A", "10", "J", "Q", "K")
    if dealer_has_peek:
        dealer_has_blackjack = Hand(cards=[dealer_up, dealer_hole], wager=0.0).is_blackjack()
    return (dealer_up, dealer_hole), dealer_has_blackjack

def dealer_play(rules: BlackjackRules, shoe: Shoe, upcard: str, hole: str) -> Hand:
    """
    Plays out the dealer hand according to H17/S17 rules.

    Args:
        rules: Dealer standing/hitting configuration.
        shoe: Shared shoe.
        upcard: Dealer's visible card.
        hole: Dealer's face-down card.

    Returns:
        Hand: Final dealer hand after drawing per rules.
    """
    dealer = Hand(cards=[upcard, hole], wager=0.0)
    # Dealer blackjack check handled by caller.
    while True:
        total = dealer.best_total()
        soft = dealer.is_soft()
        if dealer.is_busted():
            break
        if total < 17:
            dealer.add(shoe.draw(1)[0])
            continue
        if total == 17 and soft and rules.dealer_hits_soft_17:
            dealer.add(shoe.draw(1)[0])
            continue
        break
    return dealer

# ---------------------------------------------------------------------------
# LLM adapter (optional): presents a concise view to any policy/LLM
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HandView:
    """
    LLM-friendly snapshot of a player's hand and available options.

    Attributes:
        cards: Card strings in the hand.
        hard_total: Hard total value.
        soft_total: Soft total value if valid, else None.
        is_pair: Whether the first two cards form a pair.
        can_split: Whether splitting is allowed now.
        can_double: Whether doubling is allowed now.
        can_surrender: Whether surrender is allowed on this decision.
    """
    cards: List[str]
    hard_total: int
    soft_total: Optional[int]
    is_pair: bool
    can_split: bool
    can_double: bool
    can_surrender: bool

    def short_text(self) -> str:
        """Compact textual description intended for prompts/logging."""
        def soft_tag():
            return f" (soft={self.soft_total})" if self.soft_total is not None else ""
        return f"cards={self.cards}, hard={self.hard_total}{soft_tag()}, pair={self.is_pair}"

def build_hand_view(
        h: Hand,
        rules: BlackjackRules,
        dealer_up: str,
        splits_done: int,
        first_decision: bool
) -> HandView:
    """Builds a `HandView` for the given `Hand` under current table context."""
    hard, soft = h.totals()
    return HandView(
        cards=list(h.cards),
        hard_total=hard if soft is None else hard,
        soft_total=soft,
        is_pair=h.is_pair(),
        can_split=h.can_split(rules, splits_done),
        can_double=(len(h.cards) == 2) and (rules.double_after_split or not h.originated_from_split),
        can_surrender=(first_decision and rules.surrender_allowed and not h.originated_from_split),
    )

# ---------------------------------------------------------------------------
# Game loop: single round
# ---------------------------------------------------------------------------

def play_round(
        rules: BlackjackRules,
        shoe: Shoe,
        bankrolls: List[float],
        base_bets: List[float],
        players: List[LLMBlackjackPlayer],
        client: ollama.Client,
) -> Tuple[List[float], Dict]:
    """
    Plays one complete round with multiple seated players.

    Deals initial cards, handles player turns (including splits/doubles/surrender),
    plays the dealer, and settles wagers. Bankroll debits for wagers occur as
    actions (splits/doubles) are taken.

    Args:
        rules: Table rules.
        shoe: Shared shoe for the table.
        bankrolls: Per-player bankrolls (modified in-place).
        base_bets: Per-player base wagers for this round.
        players: Decision-making players exposing `.decide(...) -> Action`.
        client: Ollama client for LLM-driven players.

    Returns:
        Tuple[List[float], Dict]: Updated bankrolls and a statistics dict suitable
        for logging/analysis.

    Raises:
        RuntimeError: If any player returns an illegal action.
    """
    n = len(players)
    assert len(bankrolls) == n and len(base_bets) == n, "bankrolls/base_bets must match players length"

    # Per-player round state
    per_player = []
    for i in range(n):
        wager = base_bets[i]
        bankrolls[i] -= wager
        per_player.append({
            "wager": wager,
            "hands": [Hand(cards=shoe.draw(2), wager=wager)],  # initial single hand
            "actions": [],  # flat action log in decision order
            "splits_done": 0,
        })

    # Dealer initial
    (dealer_up, dealer_hole), dealer_has_blackjack = dealer_initial_play(shoe)

    # Immediate resolution on dealer blackjack
    if dealer_has_blackjack:
        updated_stats = {
            "dealer_blackjack": True,
            "dealer_upcard": dealer_up,
            "dealer_holecard": dealer_hole,
            "players": []
        }
        for i in range(n):
            phands = per_player[i]["hands"]
            player_has_bj = phands[0].is_blackjack()
            if player_has_bj:
                # Push: return the bet
                bankrolls[i] += per_player[i]["wager"]
                result = 0.0
            else:
                result = -per_player[i]["wager"]
            updated_stats["players"].append({
                "result": result,
                "player_totals": [h.best_total() for h in phands],
                "player_hands": phands,
                "player_actions": per_player[i]["actions"],
            })
        # Return without dealer play since hand ends on peeked BJ
        dealer_hand = Hand(cards=[dealer_up, dealer_hole], wager=0.0)
        return bankrolls, {
            **updated_stats,
            "dealer_hand": dealer_hand,
            "dealer_total": dealer_hand.best_total(),  # 21
            "dealer_upcard": dealer_up,
        }

    # Player turns (left-to-right)
    for i in range(n):
        state = per_player[i]
        hands = state["hands"]
        actions_log = state["actions"]
        splits_done = state["splits_done"]

        h_idx = 0
        while h_idx < len(hands):
            hand = hands[h_idx]
            first_decision = True
            while not hand.is_finished:
                # legal actions
                legal_actions: List["Action"] = ["hit", "stand"]
                can_double = (len(hand.cards) == 2) and (rules.double_after_split or not hand.originated_from_split)
                if can_double:
                    legal_actions.append("double")
                if hand.can_split(rules, splits_done):
                    legal_actions.append("split")
                if first_decision and rules.surrender_allowed and not hand.originated_from_split:
                    legal_actions.append("surrender")

                view = build_hand_view(hand, rules, dealer_up, splits_done, first_decision)
                action = players[i].decide(rules, view, dealer_up, legal_actions, client)
                actions_log.append(action)

                if action not in legal_actions:
                    raise RuntimeError(f"Player {i} attempted illegal action: {action}")

                if action == "surrender":
                    hand.is_surrendered = True
                    hand.is_finished = True
                    break

                if action == "double" and can_double:
                    bankrolls[i] -= hand.wager
                    hand.wager *= 2
                    hand.is_doubled = True
                    hand.add(shoe.draw(1)[0])
                    hand.is_finished = True
                    break

                if action == "split" and hand.can_split(rules, splits_done):
                    splits_done += 1
                    state["splits_done"] = splits_done
                    rank = split_rank(hand.cards[0])

                    left = Hand(cards=[hand.cards[0]], wager=hand.wager, originated_from_split=True)
                    right = Hand(cards=[hand.cards[1]], wager=hand.wager, originated_from_split=True)

                    bankrolls[i] -= hand.wager  # new wager for extra hand

                    # replace and insert
                    hands[h_idx] = left
                    hands.insert(h_idx + 1, right)

                    # deal one to each
                    hands[h_idx].add(shoe.draw(1)[0])
                    hands[h_idx + 1].add(shoe.draw(1)[0])

                    # Split Aces: one-card stand
                    if rank == "A":
                        hands[h_idx].is_split_aces = True
                        hands[h_idx + 1].is_split_aces = True
                        hands[h_idx].is_finished = True
                        hands[h_idx + 1].is_finished = True

                    # continue with left hand at same index
                    first_decision = True
                    hand = hands[h_idx]
                    continue

                if action == "hit":
                    hand.add(shoe.draw(1)[0])
                    if hand.is_busted():
                        hand.is_finished = True
                elif action == "stand":
                    hand.is_finished = True

                first_decision = False

            h_idx += 1

    # Dealer plays if any active (non-surrendered, non-busted) hand remains
    any_active = any(
        (not h.is_surrendered) and (not h.is_busted())
        for i in range(n) for h in per_player[i]["hands"]
    )
    if any_active:
        dealer_hand = dealer_play(rules, shoe, dealer_up, dealer_hole)
    else:
        dealer_hand = Hand(cards=[dealer_up, dealer_hole], wager=0.0)

    # Settlement per player
    bj_num, bj_den = rules.blackjack_pays_ratio
    table_stats: Dict = {
        "dealer_hand": dealer_hand,
        "dealer_total": dealer_hand.best_total(),
        "dealer_upcard": dealer_up,
        "players": []
    }

    for i in range(n):
        hands = per_player[i]["hands"]
        total_return = 0.0  # cash returned to bankroll after all stakes already deducted

        for h in hands:
            # per-hand settlement
            if h.is_surrendered:
                total_return += 0.5 * h.wager
            elif h.is_blackjack() and not h.originated_from_split:
                # Natural blackjack always pays 3:2 (unless dealer also had BJ, already handled in peek)
                total_return += (1.0 + bj_num / bj_den) * h.wager
            elif h.is_busted():
                total_return += 0.0
            elif dealer_hand.is_busted():
                total_return += 2.0 * h.wager
            else:
                p, d = h.best_total(), dealer_hand.best_total()
                total_return += 2.0 * h.wager if p > d else (h.wager if p == d else 0.0)

        # Do NOT add base_wager here; all stakes (base + splits + doubles) were already deducted.
        bankrolls[i] += total_return

        table_stats["players"].append({
            "result": total_return - sum(h.wager for h in hands),  # optional: net profit this round
            "player_totals": [h.best_total() for h in hands],
            "player_hands": hands,
            "player_actions": per_player[i]["actions"],
        })

    # Check if shoe needs to be re-shuffled
    return bankrolls, table_stats


# ---------------------------------------------------------------------------
# Loop over many rounds
# ---------------------------------------------------------------------------

def play_many(
        n_rounds: int = 10,
        n_players: int = 2,
        starting_bankroll: float = 1000.0,
        base_bets: List[float] = None,
        rules: Optional[BlackjackRules] = None,
        players: List[LLMBlackjackPlayer] = None,
) -> None:
    """
    Runs multiple rounds, logs each to CSV, and prints bankrolls.

    Args:
        n_rounds: Number of rounds to simulate.
        n_players: Number of seats at the table.
        starting_bankroll: Initial bankroll per player.
        base_bets: Base bet per player each round (len == n_players).
        rules: Optional table rules (defaults provided).
        players: Optional preconstructed player list; defaults to `RandomPlayer`s.

    Returns:
        None
    """
    dt_str = datetime.now().strftime("%m-%d-%Y:%H:%M:%S")
    rules = rules or BlackjackRules()
    shoe = Shoe(
        decks=rules.decks,
        penetration=0.75
    )
    client = ollama.Client()
    if players is None:
        players = [RandomPlayer() for _ in range(n_players)]  # replace with your LLMBlackjackPlayer instance

    bankrolls = [starting_bankroll for _ in range(n_players)]
    game_results = []
    for i in range(1, n_rounds + 1):
        bankrolls_before = bankrolls.copy()
        bankrolls_after, stats = play_round(rules, shoe, bankrolls, base_bets, players, client)
        # Check if shoe needs to be shuffled
        if shoe.need_shuffle:
            shoe.shuffle()
        game_results.append(stats)
        append_round_to_csv(
            table_stats=stats,
            csv_path=f"./src/data/{dt_str}.csv",
            round_meta={"rule_set": str(rules), "num_decks": 6, "penetration": 0.75},
            players_base_bets=base_bets,
            bankrolls_before=bankrolls_before,
            bankrolls_after=bankrolls_after,
        )
        print(f'Completed round: {i}', ' ' * 5, f'Bankrolls: {bankrolls_after}')
    print(f"\nFinal bankrolls: {bankrolls}")

def main():
    parser = argparse.ArgumentParser(
        description="Run multi-player blackjack simulations from the command line."
    )
    parser.add_argument("-n", "--n-players", type=int, required=True,
                        help="Number of players seated at the table.")
    parser.add_argument("-t", "--player-type", type=str, required=True,
                        help="Comma-separated list of player types/personas. "
                             "Use 'random' for RandomPlayer, otherwise treated as LLM persona. "
                             "Examples: 'basic,aggressive,conservative' or 'random' or 'basic'.")
    parser.add_argument("-r", "--rounds", type=int, required=True,
                        help="Number of rounds to play.")
    parser.add_argument("-s", "--starting-bankroll", type=float, required=True,
                        help="Starting bankroll per player.")
    # Base bet options
    parser.add_argument("--base-bet", type=float,
                        help="Single base bet to apply to all players (e.g., 10).")
    parser.add_argument("--base-bets", type=str,
                        help="Comma-separated base bets per player (e.g., '10,10,25').")

    # Optional LLM settings (used for non-random players)
    parser.add_argument("--model", type=str, default="gpt-oss:20b",
                        help="Model name for LLMBlackjackPlayer (default: gpt-oss:20b).")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature for LLMBlackjackPlayer (default: 0.1).")

    args = parser.parse_args()

    if args.n_players <= 0:
        raise ValueError("--n-players must be positive.")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive.")
    if args.starting_bankroll <= 0:
        raise ValueError("--starting-bankroll must be positive.")

    player_types = [x.strip() for x in args.player_type.split(",") if x.strip()]
    players = build_players(
        n_players=args.n_players,
        player_types=player_types,
        model=args.model,
        temperature=args.temperature
    )

    base_bets_list = [float(x.strip()) for x in args.base_bets.split(",") if x.strip()] if args.base_bets else None
    base_bets = build_base_bets(
        n_players=args.n_players,
        base_bet_value=args.base_bet,
        base_bets_list=base_bets_list
    )

    # Run
    play_many(
        n_rounds=args.rounds,
        starting_bankroll=args.starting_bankroll,
        base_bets=base_bets,
        n_players=args.n_players,
        players=players,
    )

if __name__ == "__main__":
    main()
