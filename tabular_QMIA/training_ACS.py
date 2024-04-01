# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from folktables import (
    ACSDataSource,
    ACSEmployment,
    ACSIncome,
    ACSMobility,
    ACSPublicCoverage,
    ACSTravelTime,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(description="Zipf")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--state", type=str, default="NY")
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


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

X_val, X_test_us, y_val, y_test_us = train_test_split(
    X_test, y_test, test_size=0.2, random_state=42, stratify=y_test
)

X_test = np.concatenate((X_train, X_test_us), axis=0)
y_test = np.concatenate((y_train, y_test_us), axis=0)
membership = ["in"] * X_train.shape[0] + ["out"] * X_test_us.shape[0]


def objective(trial):
    param = {
        "depth": trial.suggest_int("depth", 1, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1, log=True),
        "random_strength": trial.suggest_float("random_strength", 1, 10, log=True),
        "objective": trial.suggest_categorical(
            "objective", ["Logloss", "CrossEntropy"]
        ),
        "iterations": trial.suggest_int("iterations", 1, 1000, log=True),
    }

    # param["early_stopping_rounds"] = 5
    param["thread_count"] = 4
    param["random_seed"] = 42

    _X_train, _X_valid, _y_train, _y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=np.random.randint(0, 1000),
    )
    clf = CatBoostClassifier(**param)
    clf.fit(_X_train, _y_train, verbose=0)
    _y_pred_test = clf.predict(_X_valid, prediction_type="Probability")[:, 1]
    score = roc_auc_score(
        _y_valid,
        _y_pred_test,
    )

    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)


def detailed_objective(trial):
    param = {
        "depth": trial.suggest_int("depth", 1, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1, log=True),
        "random_strength": trial.suggest_float("random_strength", 1, 10, log=True),
        "objective": trial.suggest_categorical(
            "objective", ["Logloss", "CrossEntropy"]
        ),
        "iterations": trial.suggest_int("iterations", 1, 1000, log=True),
    }

    param["thread_count"] = 4
    param["random_seed"] = 42

    clf = CatBoostClassifier(**param)
    clf.fit(X_train, y_train, verbose=0)

    score_val = clf.predict(X_val, prediction_type="Probability")[:, 1]
    score_test = clf.predict(X_test, prediction_type="Probability")[:, 1]

    return score_val, score_test


y_score_val, y_score_test = detailed_objective(study.best_trial)

data_val = {}
data_test = {}


data_val["Task ID"] = args.task
data_val["Score"] = y_score_val
data_val["Label"] = y_val
data_val["Membership"] = ["out"] * y_val.shape[0]

data_val = pd.DataFrame(data_val)
data_val = data_val.explode(["Score", "Label", "Membership"])
data_val["Split"] = "val"

data_test["Task ID"] = args.task
data_test["Score"] = y_score_test
data_test["Label"] = y_test
data_test["Membership"] = membership

data_test = pd.DataFrame(data_test)
data_test = data_test.explode(["Score", "Label", "Membership"])
data_test["Split"] = "test"

data = pd.concat([data_val, data_test], axis=0)

data.to_csv("ACS_{}_{}_{}_cls.csv".format(args.state, args.task, args.seed))
