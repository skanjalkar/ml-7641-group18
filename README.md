# ml-7641-group18

## Project Description
This project focuses on predicting human blunders in chess games using machine learning techniques. By analyzing chess positions and their corresponding moves from a large dataset of real games, we aim to build a model that can assess the likelihood of a player making a significant mistake (blunder) in any given position.

The model leverages both classical machine learning methods and modern deep learning approaches to understand the complex patterns and decision-making processes in chess. Our dataset is sourced from Lichess, a popular open-source chess platform, and evaluated using the Stockfish chess engine to identify blunders.

## Team Members
1. Shreyas Kanjalkar
2. Harshil Vagadia
3. Lohith Kariyapla Siddalingappa
4. Jineet Hemalkumar Desai
5. Miloni Mittal

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
Download the pgn files from lichess using the following command (careful, this will download 2 files, each 30 gb worth of data).
For convenience, there is a test.pgn file already in the data folder to test how the parser works.

```bash
python3 ./scripts/download_games_torrent.py
```

## Stockfish Engine Setup

You also need to download the stockfish engine

### For Pace, please follow these exact same instructions:
- cd outside of the ml project (outside the root folder) and clone the stockfish engine
```bash
git clone git@github.com:official-stockfish/Stockfish.git
```
- cd into the Stockfish folder and build the engine
```bash
cd Stockfish/src
git checkout sf_17
make -j profile-build
```

Then move to Parsing the code section

### If working on local machine you can do the following

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

### Convert large pgn to multiple small pgn files

The dataset gives us only one pgn file with millions of games, and with the way the python-chess library operates, it is difficult to make use of multi-threading to process dataset faster.
So, we split the large pgn file into multiple smaller pgn files using pgn-extract library[https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/], which can be processed in parallel.

To split the large pgn file into multiple smaller pgn files, run the following command:
```bash
cd ./pgn-extract && make clean && make
```

```bash
./pgn-extract -#<num_games_per_file> -osplit <path_to_pgn_file>
```

For example, to split the large pgn file into multiple smaller pgn files with 100000 games per file, run the following command:
```bash
./pgn-extract -#100000 -osplit ../data/pgn-data/lichess_db_standard_rated_2024-07.pgn
```
This will create multiple pgn files in the same directory with the name 1.pgn, 2.pgn, etc. Unfortunately you can't provide an upper limit to number of files so you have to manually kill it with ctrl+c when you have enough files.
Then move the files to the data folder. WARNING: If you rerun the command it will start from the beginning again, so make sure you either use a different base pgn file or move files with different numbers than your previous ones and rename them!

```bash
mv *.pgn ../data/pgn-data/
```

### Parse the pgn files and store them into npy files

On pace, allocate the resources using the following command:
```bash
[skanjalkar3@login-ice-4 ml-7641-group18]$ salloc -N1 --ntasks-per-node=<number_of_cpu_nodes> --time=<hh:mm:ss>
[skanjalkar3@login-ice-4 ml-7641-group18]$ salloc -N1 --ntasks-per-node=32 --time=13:00:00
```

Once you have the resources, run the following command to parse the pgn files and store them into npy files according to bins:
```bash
./parse.sh
```

This will ask you the upper limit (inclusive!) on the number of files (should be equal to the number of cpu cores you have on pace) you want to parse. It expects file to be named 1.pgn 2.pgn etc.

For each game, it iterates through the moves, considering only those that meet certain criteria:
- the game has progressed beyond a minimum number of moves
- the move is randomly selected based on a probability
- there's sufficient clock time remaining.

When a move is selected, the board state is stored in X -> (8x8x17) and Y is just a binary which is blunder prediction. Z is the elo (in case we need it).
The proabability of picking the blunder move as a dataset entry is 0.25, and for that of non-blunder is 1/32. We found this observation by trial and error and found consistent results so that we have enough data points for blunder as well as non-blunder.


## Delete npy file
In order to delte the npy files, run the following command:
```bash
./scripts/delete_npy.sh
```

## TODO: Training model

## Reference Papers

Reference papers can be found in the `reference-papers` folder.
