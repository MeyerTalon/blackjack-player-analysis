# blackjack-player-analysis
An evaluation of a blackjack player's play style compared to Perfect Basic Strategy (PBS).

## System and Dependencies

This project was developed on a 2025 M4 MacBook Pro with 24GB of RAM. All usage instructions in this README are tailored for this system and MacOS.

**NOTE:** All the bash snips in this README must be run from the blackjack-player-analysis directory.

- Install and run Ollama for Mac: https://ollama.com/download

- Install python requirements:
```bash
pip install -r requirements.txt
```

- Install Latex utilities, used to generate player reports:
```bash
brew install --cask mactex
```

- Install graphviz software to visualize the blackjack game-tree:
```bash
brew install graphviz
```

## Usage

- Run the Blackjack game simulator with Ollama gpt-oss:20b agents. The following snip will play a game of 20 rounds with 2 players (one aggressive, one conservative), each with a bankroll of \$1000 and a bet size of \$50. The simulation will generate a CSV file containing the complete game history. For more information see src/blackjack_simulator/game.py.
```bash
python -m src.blackjack_simulator.game \                                
  --n-players 2 \
  --player-type conservative,aggressive \
  --rounds 20 \
  --starting-bankroll 1000 \
  --base-bet 50
```

- Run the Blackjack player analysis engine. Replace "src/data/10-09-2025:21:35:47.csv" with the file path of the game you wish to analyze. Running code will generate player reports for each player in the specified game file and store them to src/reports.
```bash
 python -m src.analysis_engine.run_analysis src/data/10-09-2025:21:35:47.csv
```

- Generate a game tree with a given player input and dealer upcard. For more granular control see src/game_tree/generate_tree.py.
```bash
python -m src.game_tree.generate_tree --hand "AH,7C" --dealer "5D"
```


## Implementation Details

