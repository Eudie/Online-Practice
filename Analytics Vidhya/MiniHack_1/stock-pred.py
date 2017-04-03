#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Eudie
"""
Here I am trying to create a trading strategy to maximize their profits in the stock market.

The task for this challenge is to predict whether the price for a particular stock at the tomorrow’s market close will
 be higher(1) or lower(0) compared to the price at today’s market close.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedKFold
from scipy.stats import randint, uniform
# reproducibility
seed = 342
np.random.seed(seed)

data = pd.read_csv("train.csv")
target = data['Outcome']

data = data.drop(['ID', 'Outcome', 'Stock_ID'], axis=1)

# train, val, target_train, target_val = train_test_split(data, target, test_size=0.25, stratify=target)
train = data
target_train = target

test = pd.read_csv("test.csv")
ids = test['ID']
test = test.drop(['ID', 'Stock_ID'], axis=1)

dtrain = xgb.DMatrix(train.values, label=target_train.values)
# dvalidation = xgb.DMatrix(val.values, label=target_val.values)
dtest = xgb.DMatrix(test.values)

print(dtrain.num_row(), dtrain.num_col())
print(dtest.num_row(), dtest.num_col())

print(np.unique(dtrain.get_label()))


params = {'objective': 'binary:logistic',
          'max_depth': 100,
          'silent': 1,
          'eta': 1,
          'n_estimators': 1,
          }


num_rounds = 1

watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_rounds, watchlist)

preds_prob = np.array(bst.predict(dtest))

# labels = np.array(dvalidation.get_label())
# preds = 1 if preds_prob > 0.5 else 0
preds = np.empty(len(preds_prob), dtype=int)
for i in range(len(preds_prob)):
    if preds_prob[i] > 0.5:
        preds[i] = 1
    else:
        preds[i] = 0


output = pd.DataFrame({'ID': ids, "Outcome":preds})


output.to_csv('submission.csv', index=False)

# correct = 0
#
# for i in range(len(preds)):
#     if labels[i] == preds[i]:
#         correct += 1
#         print('Predicted correctly: {0}/{1}'.format(correct, len(preds)))
#         print('Error: {0:.4f}'.format(1-correct/len(preds)))

# cv = StratifiedKFold(y=target_train, n_folds=10, shuffle=True, random_state=seed)
#
# params_grid = {
#                'max_depth': [3, 4, 5],
#                'n_estimators': [50, 75, 100],
#                }
#
# params_fixed = {
#                'objective': 'binary:logistic',
#                'silent': 1,
#                'learning_rate': 1
#                 }
#
# bst_grid = GridSearchCV(estimator=XGBClassifier(**params_fixed, seed=seed),
#                         param_grid=params_grid,
#                         cv=cv,
#                         scoring='accuracy')
#
# bst_grid.fit(train, target_train)
#
# print("Best accuracy obtained: {0}".format(bst_grid.best_score_))
# print("Parameters:")
# for key, value in bst_grid.best_params_.items():
#     print("\t{}: {}".format(key, value))
