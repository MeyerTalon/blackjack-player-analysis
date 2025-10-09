# Development Notes

## Implementation Plan and Acceptance Criteria

1) Synthetic Data Generation
   1) Create a blackjack game simulator using gpt-oss-20b to act as a blackjack player.
   2) Spin up 1-7 players per game, use prompt engineering to set player types (conservative, aggressive, neutral, etc)
   3) Play 20-40 games of blackjack, save the finished game data in csv format.
2) Analyze Blackjack Data
   1) Tag decisions as Complies or Deviates, and classify player styles (conservative, aggressive, neutral).
   2) EXTRA: Determine if a player is counting cards.
3) Game Tree Visualizer 
   1) Create a decision tree (JSON format) with decision paths being hit, stand, double, split, or surrender.
   2) Compute EV for each expected decision path: https://advat.blogspot.com/2012/03/decision-trees-in-optimal-blackjack.html
4) Player Report
   1) Compute house edge for a player’s betting history: compare expected value under PBS vs actual player EV. Show where the player’s mistakes increased expected losses. 
   2) Generate PDF report including:
      1) Player’s betting history (round-by-round).
      2) Deviations from PBS with explanations.
      3) Overall house edge for the session.
      4) Summary of play style (e.g., “Aggressive, frequent deviations, high expected losses”).


## Testing as I Go

- Based on a Chat-GPT prompt and Google search to verify, it looks like the best way to host gpt-oss:20b locally, M4 chip macbook pro, 24 GB RAM, is with Ollama.
- After downloading Ollama and caching the gpt-oss:20b model weights locally, initial testing of prompting gpt-oss revealed it to be quite slow (approx. 20 seconds per prompt). This means that simulating a full blackjack game (7 players, 40 rounds) would take about 1.5 hours (MUCH TOO LONG!!). To resolve this I will instead use python native async functions and the Ollama AsyncClient to run inference in parallel. 
- Running a test game prompt with 5 players took 46 secs, extrapolating a full game would take about 30 mins, an acceptable amount of time.

## Game Simulation

The game will be Classic Blackjack:

Key Rules:

- Dealer receives two cards (one face-up, one face-down).
- Dealer must hit on soft 17 (H17), stand on hard 17.
- Player blackjack pays 3:2.
- Doubling down allowed on any two cards.
- Split once.
- Surrendering allowed.
- 6 decks.
- "Blackjack" hand (A + any 10) pays out 1.5x

House Edge: ~0.5% with optimal basic strategy.

- One question that arose during development was how to classify players. Obviously there is player style; aggressive, conservative, or neutral. But then there's betting size and player strategy as well. Is this player a low roller, medium roller, or high roller (whale)? Is this player following a particular strategy like PBS? Is it likely that this player is card counting? 
- Dealer will hit on soft 17.
