"""
supervised.py - Mô hình phân lớp có giám sát: Random Forest & XGBoost
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
    **kwargs,
) -> RandomForestClassifier:
    """
    Huấn luyện Random Forest Classifier.

    Parameters
    ----------
    X_train, y_train : array-like
    n_estimators : int
    random_state : int

    Returns
    -------
    RandomForestClassifier (đã fit)
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",  # Xử lý mất cân bằng lớp
        **kwargs,
    )
    clf.fit(X_train, y_train)
    print(f"[Supervised] Random Forest đã huấn luyện ({n_estimators} trees)")
    return clf


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    random_state: int = 42,
    **kwargs,
) -> XGBClassifier:
    """
    Huấn luyện XGBoost Classifier.

    Parameters
    ----------
    X_train, y_train : array-like
    n_estimators : int
    learning_rate : float
    max_depth : int
    random_state : int

    Returns
    -------
    XGBClassifier (đã fit)
    """
    # Tính scale_pos_weight cho dữ liệu mất cân bằng
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / max(n_pos, 1)

    clf = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        **kwargs,
    )
    clf.fit(X_train, y_train)
    print(f"[Supervised] XGBoost đã huấn luyện ({n_estimators} rounds, lr={learning_rate})")
    return clf
