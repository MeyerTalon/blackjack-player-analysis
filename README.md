# blackjack-player-analysis
An evaluation of a blackjack player's play style compared to Perfect Basic Strategy (PBS). 

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
