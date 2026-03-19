"""
run_pipeline.py - Chạy toàn bộ quy trình phân tích HR Analytics

Sử dụng:
    cd bai_tap_lon
    python scripts/run_pipeline.py
"""
import sys
import os
import io

# Fix Unicode output trên Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Thêm thư mục gốc vào path để import src.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split

# Suppress sklearn warnings on Windows (they go to stderr → PowerShell treats as error)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*UndefinedMetricWarning.*")
warnings.filterwarnings("ignore", message=".*Precision is ill-defined.*")

from src.data.loader import load_data, load_config
from src.data.cleaner import HRDataCleaner
from src.features.builder import FeatureBuilder
from src.mining.association import get_attrition_rules, compare_lift_stay_vs_leave, suggest_hr_policies
from src.mining.clustering import run_kmeans, run_dbscan, run_hac, find_optimal_k, profile_clusters
from src.models.supervised import train_random_forest, train_xgboost
from src.models.semi_supervised import (
    train_semi_supervised, mask_labels,
    run_label_ratio_experiment, analyze_pseudo_label_risk,
)
from src.models.regression import train_satisfaction_regressor, check_leakage
from src.evaluation.metrics import (
    evaluate_classifier, evaluate_regressor,
    explain_with_shap, build_comparison_table, build_clustering_comparison,
)
from src.visualization.plots import (
    plot_attrition_distribution, plot_correlation_heatmap,
    plot_model_comparison, plot_feature_importance_top10,
    plot_learning_curve_semi, plot_lift_comparison,
    plot_cluster_profiles, plot_elbow_silhouette,
    plot_clustering_comparison,
)


def main():
    print("=" * 60)
    print("  HR ANALYTICS PIPELINE")
    print("=" * 60)

    # Tạo thư mục output
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    # ==================================================================
    # 1. ĐỌC CẤU HÌNH & DỮ LIỆU
    # ==================================================================
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "params.yaml")
    config = load_config(config_path)

    raw_path = os.path.join(os.path.dirname(__file__), "..", config["data"]["raw_path"])
    df = load_data(raw_path)

    # ==================================================================
    # 2. LÀM SẠCH DỮ LIỆU
    # ==================================================================
    print("\n" + "-" * 60)
    print("  BƯỚC 2: LÀM SẠCH DỮ LIỆU")
    print("-" * 60)

    cleaner = HRDataCleaner(target_col=config["data"]["target"])
    df_clean = cleaner.clean(df)

    # Lưu dữ liệu đã làm sạch
    processed_path = os.path.join(os.path.dirname(__file__), "..", config["data"]["processed_path"])
    cleaner.save_processed(df_clean, processed_path)

    # ==================================================================
    # 2b. FEATURE ENGINEERING
    # ==================================================================
    print("\n" + "-" * 60)
    print("  BƯỚC 2B: FEATURE ENGINEERING")
    print("-" * 60)

    fb = FeatureBuilder(df_clean)
    df_featured = fb.build_all()

    # ==================================================================
    # 3. EDA PLOTS
    # ==================================================================
    print("\n" + "-" * 60)
    print("  BƯỚC 3: EDA — BIỂU ĐỒ KHÁM PHÁ")
    print("-" * 60)

    plot_attrition_distribution(df_featured)
    plot_correlation_heatmap(df_featured)

    # ==================================================================
    # 4. LUẬT KẾT HỢP (ASSOCIATION RULES) — Apriori
    # ==================================================================
    print("\n" + "-" * 60)
    print("  BƯỚC 4: TÌM LUẬT KẾT HỢP (APRIORI)")
    print("-" * 60)

    df_disc = cleaner.discretize_for_mining(df_clean)
    rules = get_attrition_rules(
        df_disc,
        min_support=config["mining"]["min_support"],
        min_threshold=config["mining"]["min_threshold"],
    )

    if not rules.empty:
        print("\nTop 10 luật dẫn đến nghỉ việc (theo Lift):")
        display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        print(rules[display_cols].head(10).to_string(index=False))

        # Gợi ý chính sách HR
        suggestions = suggest_hr_policies(rules, top_n=5)

    # So sánh lift stay vs leave
    lift_comparison = compare_lift_stay_vs_leave(
        df_disc,
        min_support=config["mining"]["min_support"],
        min_threshold=1.0,
    )
    if not lift_comparison.empty:
        plot_lift_comparison(lift_comparison)
        lift_comparison.to_csv("outputs/tables/lift_comparison.csv", index=False)

    # ==================================================================
    # 5. PHÂN CỤM (CLUSTERING) — K-Means, DBSCAN, HAC
    # ==================================================================
    print("\n" + "-" * 60)
    print("  BƯỚC 5: PHÂN CỤM NHÂN VIÊN")
    print("-" * 60)

    cluster_features = config["mining"]["cluster_features"]
    cluster_features = [f for f in cluster_features if f in df_clean.columns]
    n_clusters = config["mining"]["n_clusters"]

    # Tìm K tối ưu
    from sklearn.preprocessing import StandardScaler
    X_cluster = StandardScaler().fit_transform(df_clean[cluster_features])
    optimal_result = find_optimal_k(X_cluster, k_range=range(2, 11))
    plot_elbow_silhouette(optimal_result)

    # K-Means
    df_clustered_km, km_model, sil_km = run_kmeans(
        df_clean, cluster_features, n_clusters=n_clusters,
    )
    profile_km = profile_clusters(df_clustered_km, cluster_features)

    # DBSCAN
    df_clustered_db, db_model, sil_db = run_dbscan(
        df_clean, cluster_features,
        eps=config["mining"]["dbscan_eps"],
        min_samples=config["mining"]["dbscan_min_samples"],
    )
    n_clusters_db = len(set(df_clustered_db["Cluster"])) - (1 if -1 in df_clustered_db["Cluster"].values else 0)

    # HAC
    df_clustered_hac, hac_model, sil_hac = run_hac(
        df_clean, cluster_features,
        n_clusters=n_clusters,
        linkage=config["mining"]["hac_linkage"],
    )

    # So sánh clustering
    cluster_comparison_data = {
        "K-Means": {"silhouette": sil_km, "n_clusters": n_clusters},
        "DBSCAN": {"silhouette": sil_db, "n_clusters": n_clusters_db},
        "HAC (Ward)": {"silhouette": sil_hac, "n_clusters": n_clusters},
    }
    cluster_comp_df = build_clustering_comparison(cluster_comparison_data)
    plot_clustering_comparison(cluster_comparison_data)

    plot_cluster_profiles(profile_km[cluster_features])
    profile_km.to_csv("outputs/tables/cluster_profiles.csv")

    # ==================================================================
    # 6. CHUẨN BỊ DỮ LIỆU CHO MÔ HÌNH
    # ==================================================================
    print("\n" + "-" * 60)
    print("  BƯỚC 6: CHUẨN BỊ DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH")
    print("-" * 60)

    df_encoded = cleaner.encode(df_featured)
    target = config["data"]["target"]

    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

    print(f"Train: {X_train.shape[0]} mẫu | Test: {X_test.shape[0]} mẫu")
    print(f"Tỷ lệ Attrition=1 trong train: {y_train.mean():.2%}")

    # ==================================================================
    # 7. MÔ HÌNH CÓ GIÁM SÁT (SUPERVISED)
    # ==================================================================
    print("\n" + "-" * 60)
    print("  BƯỚC 7A: RANDOM FOREST")
    print("-" * 60)

    rf_model = train_random_forest(
        X_train, y_train,
        n_estimators=config["model"]["rf_n_estimators"],
        random_state=config["model"]["random_state"],
    )
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_results = evaluate_classifier(y_test, rf_pred, rf_prob, model_name="Random Forest")
    rf_results["model_name"] = "Random Forest"

    print("-" * 60)
    print("  BƯỚC 7B: XGBOOST")
    print("-" * 60)

    xgb_model = train_xgboost(
        X_train, y_train,
        n_estimators=config["model"]["xgb_n_estimators"],
        learning_rate=config["model"]["xgb_learning_rate"],
        max_depth=config["model"]["xgb_max_depth"],
        random_state=config["model"]["random_state"],
    )
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_results = evaluate_classifier(y_test, xgb_pred, xgb_prob, model_name="XGBoost")
    xgb_results["model_name"] = "XGBoost"

    # ==================================================================
    # 8. MÔ HÌNH BÁN GIÁM SÁT (SEMI-SUPERVISED)
    # ==================================================================
    print("\n" + "-" * 60)
    print("  BƯỚC 8: SELF-TRAINING (BÁN GIÁM SÁT) — MULTI-RATIO")
    print("-" * 60)

    # Thí nghiệm tại nhiều tỷ lệ nhãn
    label_ratios = config["semi_supervised"]["label_ratios"]
    semi_curve = run_label_ratio_experiment(
        X_train.values, y_train.values,
        X_test.values, y_test.values,
        ratios=label_ratios,
        n_estimators=config["model"]["rf_n_estimators"],
        random_state=config["model"]["random_state"],
    )
    plot_learning_curve_semi(semi_curve)

    # Lưu bảng kết quả
    semi_curve["results_df"].to_csv("outputs/tables/semi_supervised_results.csv", index=False)

    # Phân tích rủi ro pseudo-label
    pseudo_risk = analyze_pseudo_label_risk(
        X_train.values, y_train.values,
        ratio=0.20,
        n_estimators=config["model"]["rf_n_estimators"],
        random_state=config["model"]["random_state"],
    )

    # Self-Training tai ratio 20% cho so sanh chung
    y_semi_20 = mask_labels(y_train.values, labeled_ratio=0.20,
                            random_state=config["model"]["random_state"])
    semi_model = train_semi_supervised(
        X_train.values, y_semi_20,
        n_estimators=config["model"]["rf_n_estimators"],
        random_state=config["model"]["random_state"],
    )
    semi_pred = semi_model.predict(X_test.values)
    semi_prob = semi_model.predict_proba(X_test.values)[:, 1]
    semi_results = evaluate_classifier(y_test, semi_pred, semi_prob,
                                       model_name="Self-Training (20% labels)")
    semi_results["model_name"] = "Self-Training (20% labels)"

    # ==================================================================
    # 9. SO SÁNH KẾT QUẢ (BẢNG + BIỂU ĐỒ)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  TỔNG HỢP KẾT QUẢ")
    print("=" * 60)

    comparison_df = build_comparison_table([rf_results, xgb_results, semi_results])
    plot_model_comparison(comparison_df)
    comparison_df.to_csv("outputs/tables/model_comparison.csv", index=False)

    # ==================================================================
    # 10. GIẢI THÍCH MÔ HÌNH (SHAP)
    # ==================================================================
    print("\n" + "-" * 60)
    print("  BƯỚC 10: GIẢI THÍCH MÔ HÌNH (SHAP)")
    print("-" * 60)

    # Giải thích mô hình tốt nhất
    best_model_name = comparison_df.loc[comparison_df["F1 (macro)"].idxmax(), "Model"]
    if "Random Forest" in best_model_name:
        shap_values, feature_imp = explain_with_shap(rf_model, X_test)
    elif "XGBoost" in best_model_name:
        shap_values, feature_imp = explain_with_shap(xgb_model, X_test)
    else:
        # SelfTrainingClassifier: try estimator_ first (new API), then base_estimator_
        try:
            base_model = semi_model.estimator_
        except AttributeError:
            base_model = getattr(semi_model, 'base_estimator_', rf_model)
        shap_values, feature_imp = explain_with_shap(base_model, X_test)

    if feature_imp is not None:
        plot_feature_importance_top10(feature_imp)
        feature_imp.to_csv("outputs/tables/feature_importance.csv", index=False)

    # ==================================================================
    # 11. HỒI QUY (REGRESSION)
    # ==================================================================
    print("\n" + "-" * 60)
    print("  BƯỚC 11: HỒI QUY MỨC ĐỘ HÀI LÒNG")
    print("-" * 60)

    reg_target = config["regression"]["target_col"]
    leakage_result = check_leakage(
        df_encoded, reg_target,
        threshold=config["regression"]["leakage_threshold"],
    )

    reg_result = train_satisfaction_regressor(
        df_encoded,
        target_col=reg_target,
        test_size=config["data"]["test_size"],
        random_state=config["model"]["random_state"],
        drop_cols=leakage_result["leaked_features"],
    )

    if reg_result:
        reg_result["results"].to_csv("outputs/tables/regression_results.csv", index=False)

    # ==================================================================
    # HOÀN TẤT
    # ==================================================================
    print("\n" + "=" * 60)
    print("  HOÀN TẤT PIPELINE!")
    print("  Kết quả đã lưu vào outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()