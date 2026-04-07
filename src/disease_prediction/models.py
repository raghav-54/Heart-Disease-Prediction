from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelArtifacts:
    logistic_model: LogisticRegression
    decision_tree_model: DecisionTreeClassifier
    tuned_random_forest_model: RandomForestClassifier
    scaler: StandardScaler


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model


def tune_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, param_grid: dict[str, list[int]]
) -> RandomForestClassifier:
    base = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_

