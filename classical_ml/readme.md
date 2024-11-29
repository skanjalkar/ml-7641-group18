# Documentation
## Using the script for training chess data on Classical ML models.

The script train_classical_ml.py will help us train classical ML models on existing npy files data processed. It processes the data, trains the Machine Learning models and provides us with train/test accuracy and cross validations scores for the data.

## Script Arguments

1. ```--data_path``` : Specify the relative path to the directory where yuor chess data is present.
2. ```--elo_list```: The specified directory would have sub directories corresponding to the ELO ranges. Specify the ELO ranges for which you would like to train classical ml models.
3. ```--model```: Specify the Machine Learning model to use. We provide three options as of now ```{'rf': RandomForest, 'lgr':Logistic Regresssion, 'svc': Support Vector Machine Classification}```
4. ```--config```: Specify the hyperparameter config json file to use for the model. We have already given some example json config files to work with
5. ```--grid_search_cv```: Set this parameter to true if you are looking to find optimal hyperparameters for your ML model. You might need to change the param_grid variable inside the script.
6. ```--cv```: Set this to true if you want 5 fold cross validation scores for the data that you want to process.
7. ```--pca```: Set this is to true if you want to use PCA for dimensionality reduction.
8. ```--pca_retain_var```: Set the value to how much variance you want to retain while performing PCA. Deafult is 0.95.

## Example Usage

Lets say we want to use a Random Forest Classifier for Blunder Prediction for Chess Data. You can execute the script by passing the following arguments. It shall give us a train-test accuracy on 80-20 split.
```
python train_classical_ml.py --elo_list=1400-1500,1500-1600,1600-1700,1700-1800,1800-1900,1900-2000 --data_path=data --config=random_forest_config.json
```
If you want to use PCA for dimensionality reduction:
```
python train_classical_ml.py --elo_list=1400-1500,1500-1600,1600-1700,1700-1800,1800-1900,1900-2000 --data_path=data --config=random_forest_config.json --pca=True
```
If you want 5 fold cross validation scores as well
```
python train_classical_ml.py --elo_list=1400-1500,1500-1600,1600-1700,1700-1800,1800-1900,1900-2000 --data_path=data --config=random_forest_config.json --cv=True
```
You can also train on other classical ML models.
```
python train_classical_ml.py --elo_list=1400-1500,1500-1600,1600-1700,1700-1800,1800-1900,1900-2000 --data_path=data --config=logistic_regression_config.json --model=lgr --cv=True
```