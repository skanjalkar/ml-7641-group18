# Imports
import numpy as np
import pandas as pd
import os
import argparse
import json

# ML model related imports.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Global variables go here.
SCRIPT_ARGS = None
DIR_LIST = []
DATA_PATH = None

#### Parse the parameters for ML model ####
###########################################
# Initialize the parser
parser = argparse.ArgumentParser(description="Add arguments for train the classical ML model")

# Define named arguments
parser.add_argument('--data_path', type=str, required=True, help="Relative path to data directory")
parser.add_argument('--elo_list', type=str, required=True, help="Comma seperated ELO ranges to process.")
parser.add_argument('--model', type=str, default='rf', help="Use rf for RandomForest, lgr for Logistic regression, svc for Support Vector Classifier")
parser.add_argument('--config', type=str, required=True, help="Specify hyperparameters to use.")
parser.add_argument('--cv', type=str, default=False, help="Set true for 5 fold cross validation")
# Parse
SCRIPT_ARGS = parser.parse_args()

############################################
DATA_PATH = SCRIPT_ARGS.data_path
DIR_LIST = SCRIPT_ARGS.elo_list.split(',')
PERFORM_CV = SCRIPT_ARGS.cv
MODEL = SCRIPT_ARGS.model
CONFIG = SCRIPT_ARGS.config
print("Elo directories that script will process: %s" % str(DIR_LIST))


# Make a compilation of the numpy files to process.
def get_files_to_process():
    """
    Helper method to fetch the numpy files corresponding to X and Y in the
    dataset.
    Returns:
        X_list, Y_list: List of numpy files in X and Y.
    """
    X_files = []
    Y_files = []
    # Iterate through each directory
    for entry in DIR_LIST:
        dir_name = os.path.join(DATA_PATH, entry)
        for filename in os.listdir(dir_name):
            # Get the X files
            if filename[0] == 'X':
                X_files.append(os.path.join(dir_name, filename))
                # Get the corresponding y file
                y_file = 'y' + filename[1:]
                Y_files.append(os.path.join(dir_name, y_file))
    return X_files, Y_files

def get_numpy_data_from_files(X_files, Y_files):
    """
    Helper method to fetch data in numpy array form
    Returns:
        X, Y: two numpy arrays.
    """
    X = None
    Y = None
    for ii in range(len(X_files)):
        if X is None:
            X = np.load(X_files[ii])
            Y = np.load(Y_files[ii])
            continue
        X = np.concatenate((X, np.load(X_files[ii])))
        Y = np.concatenate((Y, np.load(Y_files[ii])))
    return X,Y

def train_random_forest(X, Y, hyperparamter_config):
    """
    Helper method to train a random forest model using X, Y and
    hyperparameter config.
    """
    model = RandomForestClassifier(**hyperparamter_config)
    model.fit(X,Y)
    return model

def train_logistic_regression(X, Y, hyperparamter_config):
    """
    Helper method to train a random forest model using X, Y and
    hyperparameter config.
    """
    model = LogisticRegression(**hyperparamter_config)
    model.fit(X,Y)
    return model

def train_svc(X, Y, hyperparameter_config):
    """
    Helper method to train a SVC classifier.
    """
    model = SVC(**hyperparameter_config)
    model.fit(X,Y)
    return model

def get_cv_score_for_model(X, Y, hyperparameter_config):
    """
    Helper method to get cross validation scores for models.
    Returns:
        list of accuracy scores for each fold.
    """
    if MODEL == 'rf':
        model = RandomForestClassifier(**hyperparameter_config)
    elif MODEL == 'lgr':
        model = LogisticRegression(**hyperparameter_config)
    else:
        return []
    scores = cross_val_score(model, X, Y, cv=5)
    return scores

def get_accuracy(model, X, Y):
    """
    Helper method to get accuracy numbers.
    """
    y_pred = model.predict(X)
    return accuracy_score(Y, y_pred)


if __name__ == "__main__":
    X_files, Y_files = get_files_to_process()
    print("Test first 5 files from X: %s" % str(X_files[:5]))
    print("Test first 5 files from Y: %s" % str(Y_files[:5]))

    X, Y = get_numpy_data_from_files(X_files, Y_files)
    N = X.shape[0]
    X = X.reshape(N, 8*8*17)
    print("X shape: ", X.shape)
    print("Y shape: ", Y.shape)
    assert X.shape[0] == Y.shape[0], "X and Y sample mismatch"

    # Get hyperparameter config
    hyper_config = {}
    with open(CONFIG, 'r') as file:
        hyper_config = json.load(file)

    if PERFORM_CV:
        get_cv_score_for_model(X, Y, hyper_config)

    # Split the data into train and test(80% train 20% test).
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    print("Shape of X_train: ", X_train.shape)
    print("Shape of X_test: ", X_test.shape)

    if MODEL == 'rf':
        model = train_random_forest(X_train, Y_train, hyper_config)
    elif MODEL == 'lgr':
        model = train_logistic_regression(X_train, Y_train, hyper_config)
    elif MODEL == 'svc':
        model = train_svc(X_train, Y_train, hyper_config)
    else:
        raise Exception("Invalid model parameter passed: ", model)
    print("Training accuracy: ", get_accuracy(model, X_train, Y_train))
    print("Test accuracy: ", get_accuracy(model, X_test, Y_test))