# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from catboost.metrics import RMSEWithUncertainty
from folktables import (
    ACSDataSource,
    ACSEmployment,
    ACSIncome,
    ACSMobility,
    ACSPublicCoverage,
    ACSTravelTime,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(description="Zipf")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--state", type=str, default="CA")
parser.add_argument("--task", type=str, default="employment")

args = parser.parse_args()

data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
acs_data = data_source.get_data(states=[args.state], download=True)

f_task = {
    "employment": ACSEmployment,
    "income": ACSIncome,
    "coverage": ACSPublicCoverage,
    "travel": ACSTravelTime,
    "mobility": ACSMobility,
}

features, label, group = f_task[args.task].df_to_numpy(acs_data)

X, y = features, label

le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into to halves, and the first half serves as the private training set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

# Split the second half into two splits, and the first split serves as the public data,
# and the second split serves as the holdout set for evaluation
X_val, X_test_us, y_val, y_test_us = train_test_split(
    X_test, y_test, test_size=0.2, random_state=42, stratify=y_test
)

# Combine the private training set and the holdout set into a test set for evaluating
# the performance of our memberhsip inference attack
X_test = np.concatenate((X_train, X_test_us), axis=0)
y_test = np.concatenate((y_train, y_test_us), axis=0)

membership = ["in"] * X_train.shape[0] + ["out"] * X_test_us.shape[0]

# Load data that saves the probability values of individual samples
data = pd.read_csv("ACS_{}_{}_{}_cls.csv".format(args.state, args.task, args.seed))

y_score_val = data[data["Split"] == "val"]["Score"]
y_score_test = data[data["Split"] == "test"]["Score"]

y_score_test = np.array(y_score_test)
y_score_val = np.array(y_score_val)

# Transform probability values into scores for the training quantile regression model
f_score = lambda prob, l: (np.log(prob) - np.log(1 - prob)) * (2 * l - 1)

y_score_test = f_score(y_score_test, y_test)
y_score_val = f_score(y_score_val, y_val)

rng = np.random.RandomState(args.seed)


# Define the objective function for optuna
def objective(trial):
    param = {
        "depth": trial.suggest_int("depth", 1, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 1e4, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1, log=True),
        "iterations": trial.suggest_int("iterations", 1, 1000, log=True),
    }

    param["thread_count"] = 1
    param["objective"] = "RMSEWithUncertainty"
    param["posterior_sampling"] = True
    param["random_seed"] = args.seed
    eval_metric = RMSEWithUncertainty()

    # Split the data randomly into a training set for the quantile regression,
    # and a validation set for evaluation. The performance of the quantile regression model
    # on the validation set is reported to Optuna.
    _X_train, _X_valid, _y_train, _y_valid = train_test_split(
        X_val,
        y_score_val,
        test_size=0.2,
        random_state=rng.randint(0, 1000),
        stratify=y_val,
    )
    clf = CatBoostRegressor(**param)
    try:
        clf.fit(_X_train, _y_train, verbose=0)
        _y_pred_valid = clf.predict(_X_valid, prediction_type="RawFormulaVal")
        score = eval_metric.eval(label=_y_valid.T, approx=_y_pred_valid.T)
        return score
    except:
        return np.inf


# Create a study for tuning hyperparameters
study = optuna.create_study(
    direction="minimize", sampler=None, pruner=optuna.pruners.HyperbandPruner
)
study.optimize(objective, n_trials=200, n_jobs=30)


# Define the objective function that trains a quantile regression model with the best hyperparameters
def detailed_objective(trial):
    param = {
        "depth": trial.suggest_int("depth", 1, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 1e4, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1, log=True),
        "iterations": trial.suggest_int("iterations", 1, 1000, log=True),
    }

    param["thread_count"] = 1
    param["objective"] = "RMSEWithUncertainty"
    param["posterior_sampling"] = True
    param["random_seed"] = args.seed

    clf = CatBoostRegressor(**param)
    clf.fit(X_val, y_score_val, verbose=0)

    conf_test = clf.predict(X_test, prediction_type="RawFormulaVal")

    return conf_test


y_conf = detailed_objective(study.best_trial)

gaussian_pred = {}

gaussian_pred["score"] = y_score_test
gaussian_pred["mu"] = y_conf[:, 0]
gaussian_pred["log_sigma"] = y_conf[:, 1]
gaussian_pred["membership"] = membership

gaussian_pred = pd.DataFrame(gaussian_pred)

gaussian_pred.to_csv("ACS_{}_{}_qmia_{}.csv".format(args.state, args.task, args.seed))
