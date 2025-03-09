# Predicting House Prices with XGBoost Regression

This project focuses on predicting house prices using XGBoost regression. Although the project may seem simple, it involves handling skewed datasets, missing data, etc. I handled this situation by applying log transformations to certain fields to improve the model's performance.

## Data Preparation

The dataset for this project can be found on Kaggle under the name "Predicting House Prices." The training and test data are loaded from CSV files. To handle skewed fields, a log transformation is applied to specific columns such as 'LotArea', 'LotFrontage', 'YearBuilt', 'GrLivArea', 'TotalBsmtSF', and '1stFlrSF' in both the training and test datasets.

## Exploratory Data Analysis

Numerical columns are selected from the training data, and specific columns that are deemed unnecessary for the prediction task are removed. A histogram of the remaining numerical columns is plotted to provide insights into the distribution of the data.

## Model Training and Evaluation

The training data is prepared by separating the numerical columns as the features (X) and the logarithmically transformed sale prices as the target (y). The data is split into training and validation sets using a test size of 20%.

An XGBoost regression model is initialized with regularization parameters and trained on the training data. Early stopping is used to prevent overfitting, and predictions are made on the validation set. The root mean squared error (RMSE) is computed to evaluate the model's performance on the validation set.

A plot of the RMSE over the number of trees (iterations) is generated to visualize the model's learning progress.

---

This project demonstrates the use of XGBoost regression for predicting house prices. By applying log transformations to handle skewed data and training an XGBoost model with appropriate parameters, accurate predictions can be made.
