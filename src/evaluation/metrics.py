"""
metrics.py - Đánh giá mô hình: PR-AUC, F1-score, Silhouette, SHAP, MAE/RMSE
Bao gồm bảng tổng hợp so sánh mô hình.
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    auc,
    f1_score,
    silhouette_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    model_name: str = "Model",
) -> dict:
    """
    Đánh giá toàn diện mô hình phân lớp.

    Returns
    -------
    dict  {"f1": ..., "pr_auc": ..., "report": ...}
    """
    results = {}

    # F1-score (macro)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    results["f1"] = f1

    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    results["report"] = report

    # PR-AUC
    if y_prob is not None:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        results["pr_auc"] = pr_auc
    else:
        results["pr_auc"] = None

    # In kết quả
    print(f"\n{'='*50}")
    print(f"  Đánh giá: {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"  F1-score (macro): {f1:.4f}")
    if results["pr_auc"] is not None:
        print(f"  PR-AUC:           {results['pr_auc']:.4f}")
    print(f"{'='*50}\n")

    return results


def evaluate_regressor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Đánh giá mô hình hồi quy: MAE, RMSE, R².

    Returns
    -------
    dict  {"mae": ..., "rmse": ..., "r2": ...}
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  Đánh giá Regression: {model_name}")
    print(f"{'='*50}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"{'='*50}\n")

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Tính Silhouette Score cho kết quả phân cụm."""
    score = silhouette_score(X, labels)
    print(f"[Metrics] Silhouette Score: {score:.4f}")
    return score


def build_comparison_table(results_list: list[dict]) -> pd.DataFrame:
    """
    Xây dựng bảng tổng hợp so sánh tất cả mô hình.

    Parameters
    ----------
    results_list : list[dict]
        Mỗi dict: {"model_name": str, "f1": float, "pr_auc": float, ...}

    Returns
    -------
    pd.DataFrame  Bảng so sánh sắp xếp theo F1 giảm dần
    """
    rows = []
    for r in results_list:
        row = {
            "Model": r.get("model_name", "Unknown"),
            "F1 (macro)": round(r.get("f1", 0), 4),
            "PR-AUC": round(r.get("pr_auc", 0) or 0, 4),
        }
        # Thêm chi tiết nếu có
        if "report" in r and isinstance(r["report"], dict):
            report = r["report"]
            if "1" in report:
                row["Precision (class 1)"] = round(report["1"].get("precision", 0), 4)
                row["Recall (class 1)"] = round(report["1"].get("recall", 0), 4)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)

    print(f"\n{'='*70}")
    print("  BẢNG TỔNG HỢP SO SÁNH MÔ HÌNH")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"{'='*70}\n")

    return df


def build_clustering_comparison(comparison_data: dict) -> pd.DataFrame:
    """
    Bảng so sánh các phương pháp clustering.

    Parameters
    ----------
    comparison_data : dict
        {
            "KMeans": {"silhouette": 0.35, "n_clusters": 3},
            "DBSCAN": {"silhouette": 0.28, "n_clusters": 4},
            "HAC": {"silhouette": 0.33, "n_clusters": 3},
        }

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for method, data in comparison_data.items():
        rows.append({
            "Method": method,
            "Silhouette Score": round(data.get("silhouette", 0), 4),
            "Số cụm": data.get("n_clusters", "N/A"),
        })

    df = pd.DataFrame(rows).sort_values("Silhouette Score", ascending=False).reset_index(drop=True)

    print(f"\n{'='*50}")
    print("  SO SÁNH PHƯƠNG PHÁP CLUSTERING")
    print(f"{'='*50}")
    print(df.to_string(index=False))
    print(f"{'='*50}\n")

    return df


def explain_with_shap(model, X: pd.DataFrame, max_display: int = 10):
    """
    Giải thích mô hình bằng SHAP (feature importance).

    Returns
    -------
    (shap_values, feature_importance_df) hoặc (None, None)
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Nếu trả về list (binary classification), lấy class positive
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Tính trung bình absolute SHAP value cho mỗi feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "Mean |SHAP|": mean_shap,
        }).sort_values("Mean |SHAP|", ascending=False).head(max_display)

        print(f"\n[SHAP] Top {max_display} features quan trọng nhất:")
        print(feature_importance.to_string(index=False))

        return shap_values, feature_importance

    except ImportError:
        print("[SHAP] Thư viện shap chưa được cài đặt. Bỏ qua bước giải thích.")
        return None, None