##====================== fonctin d'evaluation sur la partie developpement =============

def evaluation(model):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state= RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:,1]
    
    
    dict_of_predict = {"precision":precision_score,
                       "recal":recall_score,
                       "f1_score":f1_score,
                       "class_report":classification_report,
                       "confusion_matrix":confusion_matrix,
                       "balance_accuracy":balanced_accuracy_score
                      }
    
    dict_of_proba = {"roc_auc":roc_auc_score,
                       "pr_auc":average_precision_score,
                       "brier":brier_score_loss
                      }
    for name, metric in dict_of_predict.items():
        print(f'{name :-<50} {metric(y_val, y_pred_val)}')
    for name, metric in dict_of_proba.items():
        print(f'{name :-<50} {metric(y_val, y_pred_proba)}')
    
     
    
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=cv, scoring="f1",
                                               train_sizes= np.linspace(0.1, 1, 10)
                                              )
    plt.figure(figsize = (12, 8))
    plt.plot(N, train_score.mean(axis=1), label="train_score")
    plt.plot(N, val_score.mean(axis=1), label="val_score")
    plt.legend()
    plt.show()

##====================== fonctin d'evaluation pour le modèle final =============
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import (
    precision_score, recall_score, f1_score, classification_report,
    confusion_matrix, balanced_accuracy_score, roc_auc_score,
    average_precision_score, brier_score_loss
)
import matplotlib.pyplot as plt
import numpy as np

def evaluation_model(model, X_train, y_train, X_test, y_test, random_state=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    dict_of_predict = {
        "precision": precision_score,
        "recall": recall_score,
        "f1_score": f1_score,
        "balanced_accuracy": balanced_accuracy_score
    }

    dict_of_proba = {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "brier": brier_score_loss
    }

    for name, metric in dict_of_predict.items():
        print(f"{name :-<50} {metric(y_test, y_pred):.4f}")

    for name, metric in dict_of_proba.items():
        print(f"{name :-<50} {metric(y_test, y_proba):.4f}")

    print("\nClassification report")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix")
    print(confusion_matrix(y_test, y_pred))

    N, train_score, val_score = learning_curve(
        model, X_train, y_train,
        cv=cv,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1, 10)
    )

    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label="train_score")
    plt.plot(N, val_score.mean(axis=1), label="val_score")
    plt.legend()
    plt.show()

    