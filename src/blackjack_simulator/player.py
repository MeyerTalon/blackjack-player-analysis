from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple
import asyncio
import time
import re
import random

# If you installed `ollama` >= 0.3.x:
#   pip install ollama
# The user previously used these classes, so we keep them.
from ollama import Client, AsyncClient

Action = Literal["hit", "stand", "double", "split", "surrender"]

# ------------------------------
# Game rule + hand descriptors
# ------------------------------

@dataclass(frozen=True)
class BlackjackRules:
    """Minimal rules needed for strategy prompting."""
    decks: int = 6
    dealer_hits_soft_17: bool = True  # H17=True, S17=False
    double_after_split: bool = True    # DAS
    surrender_allowed: bool = True     # Late surrender in "classic" tables
    blackjack_pays: str = "3:2"        # Shown in prompt only

@dataclass(frozen=True)
class HandView:
    """
    A lightweight, LLM-friendly description of a player's current hand.
    Provide both the raw cards and useful computed features (totals, softness).
    """
    cards: List[str]                    # e.g., ["AS", "6D"] or ["10H", "7C"]
    hard_total: int                     # e.g., 17
    soft_total: Optional[int] = None    # e.g., 17 if Ace counted as 11 (None if not applicable)
    is_pair: bool = False               # True if two-card pair (e.g., 8-8, A-A)
    can_split: bool = False             # Whether rules + bankroll + pair allow a split
    can_double: bool = True             # Whether table allows doubling now (2-card hand typically)
    can_surrender: bool = False         # Whether surrender is still available on this decision

    def short_text(self) -> str:
        def soft_tag():
            return f" (soft={self.soft_total})" if self.soft_total is not None else ""
        return f"cards={self.cards}, hard={self.hard_total}{soft_tag()}, pair={self.is_pair}"

# ------------------------------
# Built-in personas (system prompts)
# ------------------------------

PERSONAS: Dict[str, str] = {
    "aggressive": (
        "You are an aggressive blackjack player who takes bold risks to maximize winnings. "
        "Favor hitting on marginal hands, double whenever plausible, and consider nonstandard splits. "
        "Goal: short-term profit even at increased bust risk. "
        "Always reply with exactly one word: hit, stand, double, split, surrender."
    ),
    "conservative": (
        "You are a conservative blackjack player who minimizes losses and avoids busting. "
        "Favor standing on marginally safe totals, rarely double, split only when clearly beneficial (A,A and 8,8). "
        "Goal: survive longer over many hands. "
        "Always reply with exactly one word: hit, stand, double, split, surrender."
    ),
    "basic": (
        "You are a blackjack player who ALWAYS follows mathematically optimal BASIC STRATEGY for the given rules. "
        "Decide strictly by the chart logic for hard totals, soft totals, and pairs; no intuition. "
        "Always reply with exactly one word: hit, stand, double, split, or surrender when allowed by basic strategy."
    )
}

# ------------------------------
# Prompt builder
# ------------------------------

def build_decision_prompt(
    persona_name: str,
    rules: BlackjackRules,
    player_hand: HandView,
    dealer_upcard: str,
    legal_actions: Iterable[Action],
) -> str:
    """
    Compose a single-turn, instruction-complete prompt for .generate().
    We add a structured 'Game Context' block followed by a terse 'Respond with' line.
    """
    system_preamble = PERSONAS[persona_name].strip()

    legal = ", ".join(sorted({a for a in legal_actions}))
    rules_text = (
        f"Table rules:\n"
        f"- Decks: {rules.decks}\n"
        f"- Dealer {'hits' if rules.dealer_hits_soft_17 else 'stands'} on soft 17 ({'H17' if rules.dealer_hits_soft_17 else 'S17'})\n"
        f"- Double after split: {'allowed' if rules.double_after_split else 'not allowed'}\n"
        f"- Surrender: {'allowed' if rules.surrender_allowed else 'not allowed'} (late surrender)\n"
        f"- Blackjack pays: {rules.blackjack_pays_ratio}\n"
    )

    hand_text = (
        f"Game Context:\n"
        f"- Player hand: {player_hand.short_text()}\n"
        f"- Dealer upcard: {dealer_upcard}\n"
        f"- Legal actions now: {legal}\n"
    )

    # The final instruction makes parsing easy and discourages extra words.
    final = (
        "Respond with exactly one token from the legal actions set—"
        "lowercase only, no punctuation, no explanation."
    )

    return f"{system_preamble}\n\n{rules_text}\n{hand_text}\n{final}"


# ------------------------------
# Simple random  player
# ------------------------------

class RandomPlayer:
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def decide(
            self,
            rules: BlackjackRules,
            player_hand: HandView,
            dealer_upcard: str,
            legal_actions: Iterable[Action],
            client: Client) -> Action:
        legal = list(legal_actions)
        return self.rng.choice(legal)


# ------------------------------
# LLM Player class
# ------------------------------

class LLMBlackjackPlayer:
    """
    Blackjack player controlled by a local Ollama LLM (e.g., gpt-oss:20b).

    - Supports personas: 'aggressive', 'conservative', 'basic', 'random'.
    - Uses ollama.Client / AsyncClient .generate() for single-turn completions.
    - Strict output parser maps the model text to one of the allowed actions.
    """

    def __init__(
        self,
        persona: Literal["aggressive", "conservative", "basic", "random"] = "basic",
        model: str = "gpt-oss:20b",
        temperature: float = 0.2,
        max_retries: int = 2,
        timeout: Optional[float] = None,
    ) -> None:
        self.persona = persona
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout  # not all client versions support; kept for symmetry

    # --------
    # Public API
    # --------

    def decide(
            self,
            rules: BlackjackRules,
            player_hand: HandView,
            dealer_upcard: str,
            legal_actions: Iterable[Action],
            client: Client
    ) -> Action:
        """
        Synchronous version. Returns one of the legal actions.
        """
        prompt = build_decision_prompt(self.persona, rules, player_hand, dealer_upcard, legal_actions)
        text = self._call_llm_sync(prompt, client)
        return self._parse_action(text, legal_actions)

    async def decide_async(
            self,
            rules: BlackjackRules,
            player_hand: HandView,
            dealer_upcard: str,
            legal_actions: Iterable[Action],
            async_client: AsyncClient,
            sem: Optional[asyncio.Semaphore] = None,
    ) -> Action:
        """
        Async version with optional concurrency control via a semaphore.
        """
        prompt = build_decision_prompt(self.persona, rules, player_hand, dealer_upcard, legal_actions)
        if sem is None:
            return self._parse_action(await self._call_llm_async(prompt, async_client), legal_actions)
        async with sem:
            return self._parse_action(await self._call_llm_async(prompt, async_client), legal_actions)

    # ---------
    # LLM calls
    # ---------

    def _call_llm_sync(self, prompt: str, client: Client) -> str:
        last_err = None
        for _ in range(self.max_retries + 1):
            try:
                resp = client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={"temperature": self.temperature},
                )
                return (resp.get("response") or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(0.2)
        raise RuntimeError(f"Ollama sync call failed after retries: {last_err}")

    async def _call_llm_async(self, prompt: str, async_client: AsyncClient) -> str:
        last_err = None
        for _ in range(self.max_retries + 1):
            try:
                resp = await async_client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={"temperature": self.temperature},
                )
                return (resp.get("response") or "").strip()
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.2)
        raise RuntimeError(f"Ollama async call failed after retries: {last_err}")

    # --------
    # Parsing
    # --------

    @staticmethod
    def _parse_action(text: str, legal_actions: Iterable[Action]) -> Action:
        """
        Map the model's text to a valid action. We:
          - take the first word,
          - normalize to lowercase,
          - accept common synonyms (e.g., 'dd' -> 'double'),
          - fall back: if nothing matches, choose a safe legal fallback (stand if legal, else hit).
        """
        legal = {a.lower() for a in legal_actions}
        first = re.split(r"\s+", text.strip().lower())[0] if text.strip() else ""

        synonyms = {
            "h": "hit",
            "s": "stand",
            "d": "double",
            "dd": "double",
            "split": "split",
            "sp": "split",
            "su": "surrender",
            "sr": "surrender",
        }
        cand = synonyms.get(first, first)

        if cand in legal:
            return cand  # type: ignore[return-value]

        # Extra sanity: if model returned 'double' but it's illegal, prefer 'hit' if available
        if "hit" in legal and cand == "double":
            return "hit"  # type: ignore[return-value]

        # Conservative fallback order
        for choice in ("stand", "hit", "double", "split", "surrender"):
            if choice in legal:
                return choice  # type: ignore[return-value]

        # If we get here, the caller passed an empty legal set.
        raise ValueError("No legal actions available to choose from.")


