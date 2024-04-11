# CatBoost-Healthcare-Data
Apply advanced classification and regression through CatBoost to data from the healthcare/insurance sector.

## The Data
The dataset is from the healthcare sector, and it was both obfuscated and anonymized before being made available to me. 

It includes a variety of input features along with two crucial columns that represent the labels. 
- The first label is `total_cost` (for a regression problem), and it represents to the predicted total cost associated with that individual.
- The second label is `treatement` (for a binary classification problem),  and it represents whether an individual receives a particular type of treatment.

## Objectives
- Binary Classification Model: Build a supervised model to classify whether an individual is likely to receive treatment. 

- Regression Model: Develop another supervised machine learning model to predict the total cost based on the features.

## Classification and Regression with CatBoost
[CatBoost](https://catboost.ai/) is a high-performance open source library for gradient boosting on decision trees. The name comes from "Category" and "Boosting". It's designed for gradient boosting on decision trees and stands out for its capabilities in handling categorical data directly, without the need for extensive preprocessing. It includes built-in regularization techniques, such as L2 regularization on the leaf values, to prevent the model from overfitting.

More work is needed on multiple fronts including outlier detection, feature engineering, hyperparameter tuning, and model comparison.

## What about other models?
XGBoost and LightGBM require one-hot encoding of categorical features, which can cause the feature space to explode and create a memory bottleneck. Deep NNs suffer from that problem as well. Plus, they are more data-hungry than tree-based models, and with results that are far less explainable.

## To Do
- Outlier detection and feature engineering.
- More thorough hyperparameter tuning with `MLflow` https://mlflow.org/docs/latest/python_api/mlflow.catboost.html
- Further investigate the drivers of false-positives and false-negatives for the classification problem.
- Try NN models (although the presence of categorical features will make this challenging).

## Before you run any code (**IMPORTANT!**)
You will need to `unzip` the dataset file in the `data` folder.

## Results

**Metrics of "best" CatBoost classification model**
- Accuracy: 0.8851
- Precision: 0.9191
- Recall: 0.9135
- F1 Score: 0.8778
- AUC: 0.9525

**Metrics of "best" CatBoost regression model**
- With the applied `log1p` normalization:
	- RMSE: 0.2908
	- MAE: 0.1565
	- R-squared: 0.9300
	- SpearmanR: 0.9515

- On the original scale:
	- RMSE: 1880.3673
	- MAE: 806.4776
	- R-squared: 0.8252
	- SpearmanR: 0.9513