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

## Directory
`/src/parser` - folder for data processing
- `src/parser/parser.py` - code for data preprocessing
- `src/parser/config.py` - configurations for data preprocessing
- `src/parser/chess_utils.py` - code for all chess related functions
- `src/parser/game_processor.py` - code for all game based functions
- `src/parser/time_utils.py` - code for all time based functions

`/classical_ml/` - folder for classical models
- `classical_ml/logistic_regression_config.json` - configurations logistic regression model
- `classical_ml/random_forest_config.json` - configurations for random forest model
- `classical_ml/svc_config.json` - configurations for support vector machine model
- `classical_ml/train_classical_ml.py` - code for all the aforementioned models


## Virtual Environment Setup
```bash
python3 -m venv ml
source ml/bin/activate
pip install -r requirements.txt
```

## Download pgn files from lichess
```bash
python3 ./scripts/download_games_torrent.py
```
WARNING: This will download 2 files, each 30 gb worth of data. For convenience, there is a test.pgn file already in the data folder to test how the parser works.
## Stockfish Engine Setup

#### If working on Pace:

```bash
git clone git@github.com:official-stockfish/Stockfish.git
cd Stockfish/src
git checkout sf_17
make -j profile-build
```

#### If working on local machine:

```bash
sudo apt-get install stockfish
brew install stockfish
which stockfish
```

## Parsing the code

### Convert large pgn to multiple small pgn files using pgn extract library

pgn-extract library [https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/]

```bash
cd ./pgn-extract && make clean && make
./pgn-extract -#<num_games_per_file> -osplit <path_to_pgn_file>
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

This will ask you the upper limit (inclusive!) on the number of files (should be equal to the number of cpu cores you have on pace) you want to parse.

## Classical ML Models

The script train_classical_ml.py will help us train classical ML models on existing npy files data processed. It processes the data, trains the Machine Learning models and provides us with train/test accuracy and cross validations scores for the data.

Script Arguments:
1. ```--data_path``` : Specify the relative path to the directory where yuor chess data is present.
2. ```--elo_list```: The specified directory would have sub directories corresponding to the ELO ranges. Specify the ELO ranges for which you would like to train classical ml models.
3. ```--model```: Specify the Machine Learning model to use. We provide three options as of now ```{'rf': RandomForest, 'lgr':Logistic Regresssion, 'svc': Support Vector Machine Classification}```
4. ```--config```: Specify the hyperparameter config json file to use for the model. We have already given some example json config files to work with
5. ```--grid_search_cv```: Set this parameter to true if you are looking to find optimal hyperparameters for your ML model. You might need to change the param_grid variable inside the script.
6. ```--cv```: Set this to true if you want 5 fold cross validation scores for the data that you want to process.

Running Random Forest classifier:
```
python train_classical_ml.py --elo_list=1400-1500,1500-1600,1600-1700,1700-1800,1800-1900,1900-2000 --data_path=data --config=random_forest_config.json
```
Running Random Forest classifier with cross-validation:
```
python train_classical_ml.py --elo_list=1400-1500,1500-1600,1600-1700,1700-1800,1800-1900,1900-2000 --data_path=data --config=random_forest_config.json --cv=True
```
Running Logistic Regression:
```
python train_classical_ml.py --elo_list=1400-1500,1500-1600,1600-1700,1700-1800,1800-1900,1900-2000 --data_path=data --config=logistic_regression_config.json --model=lgr --cv=True
```

## Reference Papers

Reference papers can be found in the `reference-papers` folder.
