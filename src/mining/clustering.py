"""
clustering.py - Phân cụm nhân viên: K-Means, DBSCAN, HAC
Profiling cụm + HR strategy mapping
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
) -> dict:
    """
    Tìm số cụm tối ưu bằng phương pháp Elbow + Silhouette.

    Parameters
    ----------
    X : np.ndarray
        Dữ liệu đã được chuẩn hóa.
    k_range : range
        Khoảng giá trị k cần thử.

    Returns
    -------
    dict  {"inertias": [...], "silhouettes": [...], "best_k": int, "k_range": [...]}
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"[Clustering] Số cụm tối ưu (Silhouette cao nhất): k = {best_k}")

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": best_k,
    }


def run_kmeans(
    df: pd.DataFrame,
    features: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, float]:
    """
    Chạy K-Means clustering trên tập dữ liệu.

    Returns
    -------
    (df_with_cluster, kmeans_model, silhouette)
    """
    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels)

    df_result = df.copy()
    df_result["Cluster"] = labels

    print(f"[Clustering] K-Means (k={n_clusters}): Silhouette = {sil_score:.4f}")
    for c in range(n_clusters):
        cluster_data = df_result[df_result["Cluster"] == c]
        print(f"  Cụm {c}: {len(cluster_data)} nhân viên")

    return df_result, km, sil_score


def run_dbscan(
    df: pd.DataFrame,
    features: list[str],
    eps: float = 1.5,
    min_samples: int = 10,
) -> tuple[pd.DataFrame, DBSCAN, float]:
    """
    Chạy DBSCAN clustering.

    Returns
    -------
    (df_with_cluster, dbscan_model, silhouette)
    """
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    df_result = df.copy()
    df_result["Cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    if n_clusters >= 2:
        # Tính silhouette chỉ trên các điểm không phải noise
        mask = labels != -1
        sil_score = silhouette_score(X_scaled[mask], labels[mask]) if mask.sum() > 1 else -1
    else:
        sil_score = -1.0

    print(f"[Clustering] DBSCAN (eps={eps}, min_samples={min_samples}): "
          f"{n_clusters} cụm, {n_noise} noise, Silhouette = {sil_score:.4f}")

    return df_result, db, sil_score


def run_hac(
    df: pd.DataFrame,
    features: list[str],
    n_clusters: int = 3,
    linkage: str = "ward",
) -> tuple[pd.DataFrame, AgglomerativeClustering, float]:
    """
    Chạy Hierarchical Agglomerative Clustering (HAC).

    Returns
    -------
    (df_with_cluster, hac_model, silhouette)
    """
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    hac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hac.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels)

    df_result = df.copy()
    df_result["Cluster"] = labels

    print(f"[Clustering] HAC (k={n_clusters}, linkage={linkage}): Silhouette = {sil_score:.4f}")
    for c in range(n_clusters):
        cluster_data = df_result[df_result["Cluster"] == c]
        print(f"  Cụm {c}: {len(cluster_data)} nhân viên")

    return df_result, hac, sil_score


def profile_clusters(
    df_clustered: pd.DataFrame,
    features: list[str],
    target_col: str = "Attrition",
) -> pd.DataFrame:
    """
    Profiling chi tiết từng cụm + HR strategy mapping.

    Returns
    -------
    pd.DataFrame  Profile trung bình + tỷ lệ Attrition + HR strategy
    """
    profile = df_clustered.groupby("Cluster")[features].mean().round(2)

    # Thêm số lượng
    profile["Count"] = df_clustered.groupby("Cluster").size()

    # Thêm tỷ lệ Attrition nếu có
    if target_col in df_clustered.columns:
        attrition_col = df_clustered[target_col]
        if attrition_col.dtype == object:
            attrition_rate = df_clustered.groupby("Cluster")[target_col].apply(
                lambda x: (x == "Yes").mean()
            )
        else:
            attrition_rate = df_clustered.groupby("Cluster")[target_col].mean()
        profile["Attrition_Rate"] = (attrition_rate * 100).round(1)

    # HR Strategy mapping dựa trên đặc điểm cụm
    strategies = []
    for idx, row in profile.iterrows():
        strategy = _map_hr_strategy(row, features)
        strategies.append(strategy)
    profile["HR_Strategy"] = strategies

    print(f"\n{'='*60}")
    print("  PROFILING CỤM NHÂN VIÊN")
    print(f"{'='*60}")
    for idx, row in profile.iterrows():
        print(f"\n  Cụm {idx} ({row['Count']} nhân viên):")
        if "Attrition_Rate" in row:
            print(f"    Tỷ lệ nghỉ việc: {row['Attrition_Rate']}%")
        print(f"    🎯 Chiến lược HR: {row['HR_Strategy']}")

    return profile


def _map_hr_strategy(row, features):
    """Ánh xạ đặc điểm cụm → chiến lược HR."""
    parts = []

    # Tuổi
    if "Age" in features:
        if row.get("Age", 35) < 30:
            parts.append("Nhóm trẻ → cần career path rõ ràng, đào tạo")
        elif row.get("Age", 35) > 45:
            parts.append("Nhóm senior → mentor program, giữ chân bằng vai trò lãnh đạo")

    # Thu nhập
    if "MonthlyIncome" in features:
        if row.get("MonthlyIncome", 6000) < 4000:
            parts.append("Thu nhập thấp → xem xét điều chỉnh lương")

    # Thâm niên
    if "YearsAtCompany" in features:
        if row.get("YearsAtCompany", 5) < 3:
            parts.append("Thâm niên ngắn → cải thiện onboarding, engagement")
        elif row.get("YearsAtCompany", 5) > 10:
            parts.append("Trung thành → thưởng loyalty, cơ hội thăng tiến")

    # Hài lòng
    if "JobSatisfaction" in features:
        if row.get("JobSatisfaction", 3) < 2.5:
            parts.append("Hài lòng thấp → cải thiện môi trường, lắng nghe feedback")

    # Attrition rate
    attrition_rate = row.get("Attrition_Rate", 0)
    if attrition_rate > 25:
        parts.append("⚠️ RỦI RO CAO - ưu tiên can thiệp ngay")
    elif attrition_rate < 10:
        parts.append("✅ Ổn định - duy trì chính sách hiện tại")

    return "; ".join(parts) if parts else "Cần phân tích thêm"