import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings('ignore')

# Defining columns and reading data
col = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
df = pd.read_csv('auto-mpg.data', names=col, na_values="?", comment='\t', sep=" ", skipinitialspace=True)
data = df.copy()
# print(data.sample(10))


# creating a training and test split

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['Cylinders']):
    # print(test_index)
    start_train_set = data.loc[train_index]
    start_test_set = data.loc[test_index]

# Exploratory data Analysis

# print(data.info())
# print(data.isnull().sum())
# print(data.describe())

# sns.boxplot(x=data['Horsepower'])
# plt.show()


# Category distribution
# print(data['Cylinders'].value_counts())
# distribution in %
# print(data['Cylinders'].value_counts() / len(data))

# print(data['Origin'].value_counts())

# finding intuition of potential correlation
# sns.pairplot(data[['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']])
# plt.show()

# Segregating Target and feature variables

data = start_train_set.drop("MPG", axis=1)
data_labels = start_train_set['MPG'].copy()


# Preprocessing the original columns
def preprocess_origin_clos(df):
    df["Origin"] = df['Origin'].map({1: 'India', 2: 'USA', 3: 'Germany'})
    return df


'''
data_tr = preprocess_origin_clos(data)
# print(data_tr.head())


# print(data_tr.info())
# isolating the categorical variable which is origin column
data_cat = data_tr[['Origin']]
# print(data_cat.head())

# one hot encoding the categorical variable
cat_encoder = OneHotEncoder()
data_cat_oh = cat_encoder.fit_transform(data_cat)  # returns a sparse matrix
# print(data_cat_oh)

# print(cat_encoder.categories_)


# Handling the missing values using SimpleImputer
# segregating the numerical columns
num_data = data.iloc[:, :-1]
# print(num_data.info())

# handling missing values
imputer = SimpleImputer(strategy='median')
imputer.fit(num_data)
# print(imputer.statistics_)

# imputing the missing values by transforming the dataframe
x = imputer.transform(num_data)
# print(x)

# converting the @D array back into a dataframe
data_tr = pd.DataFrame(x, columns=num_data.columns, index=num_data.index)
# print(data_tr.info())
'''

# adding attributes using BaseEstimator and Transformer
# print(num_data.head())
acc_x, hpower_x, cyl_x = 4, 2, 0


class CustomAttributes(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True):  # no arguments
        self.acc_on_power = acc_on_power

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        acc_on_cyl = X[:, acc_x] / X[:, hpower_x]
        if self.acc_on_power:
            acc_on_power = X[:, acc_x] / X[:, hpower_x]
            return np.c_[X, acc_on_power, acc_on_cyl]
        return np.c_[X, acc_on_cyl]


# attr_adder = CustomAttributes(acc_on_power=True)
# data_tr_extra_attr = attr_adder.transform(data_tr.values)


# print(data_tr_extra_attr[0])


# creating a pipeline of tasks

'''
Function to process numerical tranformation 
Arguments: 
data: original dataframe
returns :
num_attrs : numerical attributes 
num_pipeline :numerical pipeline object
'''


def num_pipeline_transformer(data):
    numerics = ['float64', 'int64']
    num_attrs = data.select_dtypes(include=numerics)

    # pipeline for numerical attributes
    # imputing adding attributes --> scale them
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attr_adder', CustomAttributes()),
        ('std_scaler', StandardScaler())
    ])
    return num_attrs, num_pipeline


'''
Complete transformation pipeline for both numerical and categorical data.
Argument:
data: original dataframe 
returns:
prepared_data: transformed data, ready to use 
'''


def pipeline_transformer(data):
    # Transforming Numerical and Categorical Attributes
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    cat_attrs = ["Origin"]
    # Complete pipeline to transform
    # both numerical and categorical attributes

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
        ("cat", OneHotEncoder(), cat_attrs),
    ])

    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data


# from raw data to processed data
preprocessed_df = preprocess_origin_clos(data)
prepared_data = pipeline_transformer(preprocessed_df)
# print(prepared_data[0])

# Selecting and Training models
'''
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(prepared_data, data_labels)
# testing the predictions
sample_data = data.iloc[: 5]
sample_labels = data_labels.iloc[: 5]

sample_data_prepared = pipeline_transformer(sample_data)
# print("Prediction of samples: ", lin_reg.predict(sample_data_prepared))
# print("Actual Labels of samples: ", list(sample_labels))


# Mean Squared error it tell how many error in the prediction
# checking Liner Regression predictions
mpg_predictions = lin_reg.predict(prepared_data)
lin_mse = mean_squared_error(data_labels, mpg_predictions)
lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)


# Decision Tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(prepared_data, data_labels)
# checking Decision tree regressor predictions
mpg_predictions = tree_reg.predict(prepared_data)
tree_mse = mean_squared_error(data_labels, mpg_predictions)
tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)


# Model Evaluation using cross validation
# it creates K folds and gives result is an array containing the K evaluation scores:
# Decision tree evaluation
scores = cross_val_score(tree_reg, prepared_data, data_labels, scoring="neg_mean_squared_error", cv=10)
# -scores is used bcz it neg_mean_squared_error gives negative value to make it positive we use -
tree_reg_rmse_scores = np.sqrt(-scores)
# print(tree_reg_rmse_scores)
# print(tree_reg_rmse_scores.mean())

# Linear Regression evaluation
scores = cross_val_score(lin_reg, prepared_data, data_labels, scoring="neg_mean_squared_error", cv=10)
lin_reg_rmse_scores = np.sqrt(-scores)
# print(tree_reg_rmse_scores)
# print(lin_reg_rmse_scores.mean())


# Support Vector Machine Regressor SVM
svm_reg = SVR(kernel='linear')
svm_reg.fit(prepared_data, data_labels)
# SVM evaluation
scores = cross_val_score(svm_reg, prepared_data, data_labels, scoring="neg_mean_squared_error", cv=10)
svm_reg_scores = np.sqrt(-scores)
# print(svm_reg_scores)
# print(svm_reg_scores.mean())
'''
# Random Forest Regressor
ran_fros = RandomForestRegressor()
ran_fros.fit(prepared_data, data_labels)
# Random Forest Regressor evaluation
scores = cross_val_score(ran_fros, prepared_data, data_labels, scoring="neg_mean_squared_error", cv=10)
ran_fros_scores = np.sqrt(-scores)
# print(ran_fros_scores)
# print(ran_fros_scores.mean())

# Finding which set of prameter works the best for Random Forest Regressor to improve the performance
# Hyper parameter Tuning using GridSearchCV
pram_grid = [{
    'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]
     }]

grid_search = GridSearchCV(ran_fros, pram_grid, scoring='neg_mean_squared_error', return_train_score=True, cv=10)
grid_search.fit(prepared_data, data_labels)
# print(grid_search.best_params_)

# checking parameters scores so we can club them together
cv_score = grid_search.cv_results_
for mean_score, param in zip(cv_score['mean_test_score'], cv_score['params']):
    # print(np.sqrt(-mean_score), param)
    pass

# checking Features importance
feature_importance = grid_search.best_estimator_.feature_importances_
# print(feature_importance)

extra_attrs = ['acc_on_power', 'acc_on_cly']
numerics = ['float64', 'int64']
num_attrs = list(data.select_dtypes(include=numerics))
attrs = num_attrs + extra_attrs
# print(sorted(zip(attrs, feature_importance), reverse=True))


# Evaluating the entire system on test data
# Final model

final_model = grid_search.best_estimator_

X_test = start_test_set.drop("MPG", axis=1)
Y_test = start_test_set['MPG'].copy()

X_test_preprocessed = preprocess_origin_clos(X_test)
X_test_prepared = pipeline_transformer(X_test_preprocessed)

final_predictions = final_model.predict(X_test_preprocessed)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# print(final_rmse)


def predict_mpg(config, model):
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    preproc_df = preprocess_origin_clos(df)
    prepared_df = pipeline_transformer(preproc_df)
    print(preproc_df)
    y_pred = model.predict(prepared_df)
    return y_pred


# checking it on random sample
vehicle_config = {
    'Cylinders': [4, 6, 8],
    'Displacement': [1550., 160.0, 165.5],
    'Horsepower': [93.0, 130.0, 98.0],
    'Weight': [2500.0, 3150.0, 2660.0],
    'Acceleration': [15.0, 14.0, 16.0],
    'Model Year': [81, 80, 78],
    'Origin': [3, 2, 1]
}

print(predict_mpg(vehicle_config, final_model))
