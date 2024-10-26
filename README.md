# ml-7641-group18

## Team Members
Shreyas Kanjalkar
Harshil Vagadia
Lohith Kariyapla Siddalingappa
Jineet Hemalkumar Desai
Miloni Mittal

## Virtual Environment Setup
1. Create a virtual environment using the following command:
```bash
python3 -m venv ml
```

2. Activate the virtual environment using the following command:
```bash
source ml/bin/activate
```

3. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

4. Deactivate the virtual environment using the following command:
```bash
deactivate
```

## Download pgn files from lichess
1. Download the pgn files from lichess using the following command (careful, this will download 2 files, each 30 gb worth of data).
For convenience, there is a test.pgn file already in the data folder to test how the parser works.
```bash
python3 ./scripts/download_games_torrent.sh
```

## You also need to download the stockfish engine
```bash
sudo apt-get install stockfish
```

For mac:
```bash
brew install stockfish
```

Find where your stockfish is installed and update the path in the config.py file.
```bash
which stockfish
```

## Parsing the code

For each game, it iterates through the moves, considering only those that meet certain criteria:
- the game has progressed beyond a minimum number of moves
- the move is randomly selected based on a probability
- there's sufficient clock time remaining.

When a move is selected, the board state is stored in X -> (8x8x17) and Y is just a binary which is blunder prediction. Z is the elo
The ratio of non-blunder to blunder is 2:1

1. Run the following command to parse the pgn files and store them into npy files according to bins:
```bash
python3 src/parser/parser.py
```
