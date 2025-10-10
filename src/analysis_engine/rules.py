from dataclasses import dataclass

@dataclass
class Rules:
    decks: int = 6
    h17: bool = True
    das: bool = True
    surrender: bool = True  # Late surrender

# Baseline PBS house edge for 6D H17 DAS LS (approx.)
PBS_BASELINE_EDGE = 0.0065  # 0.65%
