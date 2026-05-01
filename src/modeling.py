import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

##===================== Construction et récupération du meilleure modéle apres optimisation =============

def build_logistic_pipeline(preprocessor, random_state=42):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ))
    ])


def model_optimal(
    preprocessor,
    X_train,
    y_train,
    random_state=42,
    scoring="f1"
):
    pipeline = build_logistic_pipeline(preprocessor, random_state=random_state)

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ["lbfgs", "liblinear"],
        "model__max_iter": [1000, 2000],
        "model__class_weight": ["balanced"]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    grid = GridSearchCV(pipeline, param_grid, scoring="f1", cv=cv, n_jobs=-1
                       )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)

##========================== 