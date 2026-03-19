"""
semi_supervised.py - Mô hình bán giám sát: Self-Training Classifier
Giả lập kịch bản thiếu nhãn, so sánh supervised vs semi-supervised tại nhiều tỷ lệ nhãn.
Phân tích rủi ro pseudo-label.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, auc, classification_report


def mask_labels(
    y: np.ndarray,
    labeled_ratio: float = 0.2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Giả lập dữ liệu thiếu nhãn: giữ lại `labeled_ratio` nhãn,
    phần còn lại đặt thành -1 (unlabeled).

    Parameters
    ----------
    y : array-like
        Nhãn gốc (0/1).
    labeled_ratio : float
        Tỷ lệ dữ liệu ĐƯỢC GIỮ nhãn (VD: 0.20 = giữ 20%, ẩn 80%).
    random_state : int

    Returns
    -------
    np.ndarray   Nhãn mới với -1 thay cho nhãn bị ẩn.
    """
    rng = np.random.RandomState(random_state)
    y_semi = np.array(y, dtype=int).copy()

    n = len(y_semi)
    n_keep = max(int(n * labeled_ratio), 2)  # Giữ tối thiểu 2 mẫu

    # Stratified: giữ tỷ lệ cân bằng giữa các lớp
    keep_indices = []
    for cls in np.unique(y_semi):
        cls_indices = np.where(y_semi == cls)[0]
        n_cls_keep = max(int(len(cls_indices) * labeled_ratio), 1)
        chosen = rng.choice(cls_indices, size=n_cls_keep, replace=False)
        keep_indices.extend(chosen.tolist())

    mask = np.ones(n, dtype=bool)  # True = ẩn nhãn
    mask[keep_indices] = False
    y_semi[mask] = -1

    n_labeled = np.sum(y_semi != -1)
    n_pos = np.sum(y_semi == 1)
    print(f"[Semi-supervised] Giữ lại {n_labeled}/{n} nhãn "
          f"({n_labeled / n * 100:.1f}%), trong đó {n_pos} positive")
    return y_semi


def train_semi_supervised(
    X_train: np.ndarray,
    y_semi: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
    **kwargs,
) -> SelfTrainingClassifier:
    """
    Huấn luyện Self-Training Classifier (bán giám sát).

    Returns
    -------
    SelfTrainingClassifier (đã fit)
    """
    base_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
    )
    self_training_clf = SelfTrainingClassifier(
        estimator=base_clf,
        threshold=0.7,        # Giảm threshold để dễ gán pseudo-label hơn
        max_iter=15,           # Tăng vòng lặp
        criterion="threshold",
    )
    self_training_clf.fit(X_train, y_semi)

    print(f"[Semi-supervised] Self-Training đã huấn luyện "
          f"(base: RF {n_estimators} trees)")
    return self_training_clf


def run_label_ratio_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ratios: list = None,
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict:
    """
    Thí nghiệm bán giám sát tại nhiều tỷ lệ nhãn.
    So sánh Supervised-only vs Self-Training tại mỗi ratio.

    Parameters
    ----------
    ratios : list[float]
        Danh sách tỷ lệ nhãn giữ lại (VD: [0.05, 0.10, 0.20, 0.30]).

    Returns
    -------
    dict  {
        "ratios": [...],
        "supervised_f1": [...], "supervised_pr_auc": [...],
        "semi_f1": [...], "semi_pr_auc": [...],
        "details": [...]
    }
    """
    if ratios is None:
        ratios = [0.05, 0.10, 0.20, 0.30]

    supervised_f1 = []
    supervised_pr_auc = []
    semi_f1 = []
    semi_pr_auc = []
    details = []

    print(f"\n{'='*60}")
    print("  THÍ NGHIỆM BÁN GIÁM SÁT: SUPERVISED vs SELF-TRAINING")
    print(f"{'='*60}")

    y_train_arr = np.array(y_train, dtype=int)
    y_test_arr = np.array(y_test, dtype=int)

    for ratio in ratios:
        print(f"\n--- Tỷ lệ nhãn: {ratio*100:.0f}% ---")

        # Tạo nhãn bị ẩn (stratified)
        y_semi = mask_labels(y_train_arr, labeled_ratio=ratio, random_state=random_state)
        labeled_mask = y_semi != -1

        # ---- Supervised-only (chỉ dùng dữ liệu có nhãn) ----
        X_labeled = X_train[labeled_mask]
        y_labeled = y_train_arr[labeled_mask]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            sup_clf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight="balanced",
            )
            sup_clf.fit(X_labeled, y_labeled)
            sup_pred = sup_clf.predict(X_test)
            sup_prob = sup_clf.predict_proba(X_test)[:, 1]

            sup_f1 = f1_score(y_test_arr, sup_pred, average="macro", zero_division=0)
            precision_arr, recall_arr, _ = precision_recall_curve(y_test_arr, sup_prob)
            sup_prauc = auc(recall_arr, precision_arr)

        supervised_f1.append(sup_f1)
        supervised_pr_auc.append(sup_prauc)

        # ---- Self-Training (bán giám sát) ----
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            semi_clf = train_semi_supervised(
                X_train, y_semi,
                n_estimators=n_estimators,
                random_state=random_state,
            )
            semi_pred = semi_clf.predict(X_test)
            semi_prob = semi_clf.predict_proba(X_test)[:, 1]

            s_f1 = f1_score(y_test_arr, semi_pred, average="macro", zero_division=0)
            precision_arr, recall_arr, _ = precision_recall_curve(y_test_arr, semi_prob)
            s_prauc = auc(recall_arr, precision_arr)

        semi_f1.append(s_f1)
        semi_pr_auc.append(s_prauc)

        detail = {
            "ratio": ratio,
            "n_labeled": int(labeled_mask.sum()),
            "supervised_f1": round(sup_f1, 4),
            "supervised_pr_auc": round(sup_prauc, 4),
            "semi_f1": round(s_f1, 4),
            "semi_pr_auc": round(s_prauc, 4),
            "f1_improvement": round(s_f1 - sup_f1, 4),
            "prauc_improvement": round(s_prauc - sup_prauc, 4),
        }
        details.append(detail)

        print(f"  Supervised: F1={sup_f1:.4f}, PR-AUC={sup_prauc:.4f}")
        print(f"  Semi-sup:   F1={s_f1:.4f}, PR-AUC={s_prauc:.4f}")
        delta_f1 = detail['f1_improvement']
        delta_prauc = detail['prauc_improvement']
        print(f"  Cai thien:  dF1={delta_f1:+.4f}, dPR-AUC={delta_prauc:+.4f}")

    # Tong hop bang
    results_df = pd.DataFrame(details)
    print(f"\n{'='*60}")
    print("  TONG HOP KET QUA")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    return {
        "ratios": ratios,
        "supervised_f1": supervised_f1,
        "supervised_pr_auc": supervised_pr_auc,
        "semi_f1": semi_f1,
        "semi_pr_auc": semi_pr_auc,
        "details": details,
        "results_df": results_df,
    }


def analyze_pseudo_label_risk(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ratio: float = 0.20,
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict:
    """
    Phân tích rủi ro gán nhãn sai (pseudo-label) và tác dong chinh sach.

    So sánh pseudo labels (do Self-Training gán) với nhãn thực
    de danh gia chat luong pseudo-label.

    Returns
    -------
    dict  {
        "total_pseudo": int,
        "correct_pseudo": int,
        "accuracy": float,
        "false_positive_rate": float,
        "false_negative_rate": float,
        "policy_risk_analysis": str,
    }
    """
    y_arr = np.array(y_train, dtype=int).copy()
    y_semi = mask_labels(y_arr, labeled_ratio=ratio, random_state=random_state)

    unlabeled_mask = y_semi == -1
    true_labels_hidden = y_arr[unlabeled_mask]

    # Huan luyen self-training
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        semi_clf = train_semi_supervised(
            X_train, y_semi,
            n_estimators=n_estimators,
            random_state=random_state,
        )

    # Lay pseudo labels = du doan tren du lieu unlabeled
    pseudo_labels = semi_clf.predict(X_train[unlabeled_mask])

    # So sanh voi nhan thuc
    total_pseudo = len(pseudo_labels)
    correct = int((pseudo_labels == true_labels_hidden).sum())
    accuracy = correct / max(total_pseudo, 1)

    # False positive: gan nhan "nghi viec" nhung thuc te "o lai"
    fp = int(((pseudo_labels == 1) & (true_labels_hidden == 0)).sum())
    fp_denom = max(int((true_labels_hidden == 0).sum()), 1)
    fp_rate = fp / fp_denom

    # False negative: gan nhan "o lai" nhung thuc te "nghi viec"
    fn = int(((pseudo_labels == 0) & (true_labels_hidden == 1)).sum())
    fn_denom = max(int((true_labels_hidden == 1).sum()), 1)
    fn_rate = fn / fn_denom

    # Phan tich rui ro chinh sach
    risk_analysis = []
    risk_analysis.append(f"Pseudo-label accuracy: {accuracy:.1%} ({correct}/{total_pseudo})")

    if fp_rate > 0.2:
        risk_analysis.append(
            f"WARNING: FALSE POSITIVE cao ({fp_rate:.1%}): Nhieu NV bi gan nhan 'se nghi' "
            f"nhung thuc te o lai -> lang phi ngan sach can thiep"
        )
    if fn_rate > 0.3:
        risk_analysis.append(
            f"ERROR: FALSE NEGATIVE cao ({fn_rate:.1%}): Nhieu NV 'se nghi' bi bo sot "
            f"-> mat nhan tai do khong can thiep kip thoi"
        )

    if accuracy > 0.85:
        risk_analysis.append(
            "OK: Pseudo-labels dang tin cay -> co the dung de mo rong du lieu huan luyen"
        )
    elif accuracy > 0.70:
        risk_analysis.append(
            "WARN: Pseudo-labels chap nhan duoc -> nen ket hop voi kiem tra thu cong"
        )
    else:
        risk_analysis.append(
            "ERROR: Pseudo-labels kem -> KHONG nen trien khai, can thu thap them nhan thuc"
        )

    policy_risk = "\n  ".join(risk_analysis)
    print(f"\n{'='*60}")
    print("  PHAN TICH RUI RO PSEUDO-LABEL")
    print(f"{'='*60}")
    print(f"  {policy_risk}")

    return {
        "total_pseudo": total_pseudo,
        "correct_pseudo": correct,
        "accuracy": round(accuracy, 4),
        "false_positive_rate": round(fp_rate, 4),
        "false_negative_rate": round(fn_rate, 4),
        "false_positives": fp,
        "false_negatives": fn,
        "policy_risk_analysis": policy_risk,
    }
