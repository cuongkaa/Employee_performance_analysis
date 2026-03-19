"""
plots.py - Module vẽ biểu đồ dùng chung cho HR Analytics
Tất cả hàm đều save figure vào output_dir.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Dùng backend không cần GUI
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình mặc định
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})
sns.set_style("whitegrid")

DEFAULT_OUTPUT = "outputs/figures"


def _save(fig, filename, output_dir=DEFAULT_OUTPUT):
    """Lưu figure và đóng."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Plot] Đã lưu: {path}")
    return path


# ==================================================================
# 1. EDA Plots
# ==================================================================

def plot_attrition_distribution(df, target="Attrition", output_dir=DEFAULT_OUTPUT):
    """Biểu đồ phân bố Attrition (Yes/No)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count plot
    counts = df[target].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Phân bố Attrition")
    axes[0].set_ylabel("Số lượng")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=colors, startangle=90, explode=(0.05, 0))
    axes[1].set_title("Tỷ lệ Attrition")

    fig.suptitle("Phân bố nghỉ việc (Attrition)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "attrition_distribution.png", output_dir)


def plot_attrition_by_factor(df, factor, target="Attrition", output_dir=DEFAULT_OUTPUT):
    """Biểu đồ tỷ lệ Attrition theo một yếu tố cụ thể."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ct = pd.crosstab(df[factor], df[target], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"],
            edgecolor="black", ax=ax)
    ax.set_title(f"Tỷ lệ nghỉ việc theo {factor}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_xlabel(factor)
    ax.legend(title=target)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return _save(fig, f"attrition_by_{factor.lower()}.png", output_dir)


def plot_correlation_heatmap(df, output_dir=DEFAULT_OUTPUT):
    """Ma trận tương quan (chỉ cột số)."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Ma trận tương quan", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "correlation_heatmap.png", output_dir)


def plot_numeric_distributions(df, target="Attrition", output_dir=DEFAULT_OUTPUT):
    """Box plots cho các biến số quan trọng theo Attrition."""
    important_cols = [
        "Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
        "DistanceFromHome", "JobSatisfaction", "EnvironmentSatisfaction",
    ]
    cols = [c for c in important_cols if c in df.columns]

    n = len(cols)
    nrows = (n + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(x=target, y=col, data=df, ax=axes[i],
                    palette=["#2ecc71", "#e74c3c"])
        axes[i].set_title(f"{col} theo {target}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Phân bố các biến số theo Attrition", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "numeric_distributions.png", output_dir)


# ==================================================================
# 2. Clustering Plots
# ==================================================================

def plot_cluster_profiles(profile_df, output_dir=DEFAULT_OUTPUT):
    """Biểu đồ radar/bar cho profile từng cụm."""
    n_clusters = len(profile_df)
    features = profile_df.columns.tolist()

    # Normalize để so sánh
    profile_norm = (profile_df - profile_df.min()) / (profile_df.max() - profile_df.min() + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(features))
    width = 0.8 / n_clusters
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        ax.bar(x + i * width, profile_norm.iloc[i], width,
               label=f"Cụm {i}", color=colors[i], edgecolor="black", alpha=0.8)

    ax.set_xticks(x + width * (n_clusters - 1) / 2)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_ylabel("Giá trị chuẩn hóa")
    ax.set_title("Profile các cụm nhân viên (chuẩn hóa)", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "cluster_profiles.png", output_dir)


def plot_elbow_silhouette(eval_result, output_dir=DEFAULT_OUTPUT):
    """Biểu đồ Elbow + Silhouette cho chọn K."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    k_range = eval_result["k_range"]

    ax1.plot(k_range, eval_result["inertias"], "bo-", linewidth=2)
    ax1.set_xlabel("Số cụm (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Phương pháp Elbow")

    ax2.plot(k_range, eval_result["silhouettes"], "rs-", linewidth=2)
    ax2.set_xlabel("Số cụm (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score theo k")
    best_k = eval_result["best_k"]
    idx = k_range.index(best_k)
    ax2.annotate(f"Best k={best_k}", xy=(best_k, eval_result["silhouettes"][idx]),
                 fontsize=12, fontweight="bold", color="red",
                 arrowprops=dict(arrowstyle="->", color="red"))

    fig.suptitle("Chọn số cụm tối ưu", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "elbow_silhouette.png", output_dir)


# ==================================================================
# 3. Model Comparison Plots
# ==================================================================

def plot_model_comparison(comparison_df, output_dir=DEFAULT_OUTPUT):
    """Biểu đồ so sánh các mô hình (F1 + PR-AUC)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = comparison_df["Model"].values
    x = np.arange(len(models))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    # F1 Score
    axes[0].barh(x, comparison_df["F1 (macro)"].values, color=colors, edgecolor="black")
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(models)
    axes[0].set_xlabel("F1 Score (macro)")
    axes[0].set_title("So sánh F1 Score")
    for i, v in enumerate(comparison_df["F1 (macro)"].values):
        axes[0].text(v + 0.005, i, f"{v:.4f}", va="center", fontweight="bold")

    # PR-AUC
    pr_auc = comparison_df["PR-AUC"].values
    axes[1].barh(x, pr_auc, color=colors, edgecolor="black")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(models)
    axes[1].set_xlabel("PR-AUC")
    axes[1].set_title("So sánh PR-AUC")
    for i, v in enumerate(pr_auc):
        axes[1].text(v + 0.005, i, f"{v:.4f}", va="center", fontweight="bold")

    fig.suptitle("So sánh hiệu suất các mô hình", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "model_comparison.png", output_dir)


def plot_feature_importance_top10(importance_df, output_dir=DEFAULT_OUTPUT):
    """Top 10 features quan trọng nhất (SHAP hoặc feature importance)."""
    top10 = importance_df.head(10)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top10)))
    ax.barh(range(len(top10)), top10.iloc[:, 1].values, color=colors, edgecolor="black")
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10.iloc[:, 0].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Features Quan Trọng Nhất", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "feature_importance_top10.png", output_dir)


# ==================================================================
# 4. Semi-supervised Learning Curve
# ==================================================================

def plot_learning_curve_semi(curve_data, output_dir=DEFAULT_OUTPUT):
    """
    Learning curve: % nhãn vs PR-AUC/F1.

    Parameters
    ----------
    curve_data : dict
        {
            "ratios": [0.05, 0.10, 0.20, 0.30],
            "supervised_f1": [...],
            "supervised_pr_auc": [...],
            "semi_f1": [...],
            "semi_pr_auc": [...],
        }
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ratios_pct = [r * 100 for r in curve_data["ratios"]]

    # F1 comparison
    ax1.plot(ratios_pct, curve_data["supervised_f1"], "o-",
             label="Supervised-only", color="#3498db", linewidth=2, markersize=8)
    ax1.plot(ratios_pct, curve_data["semi_f1"], "s-",
             label="Self-Training", color="#e74c3c", linewidth=2, markersize=8)
    ax1.set_xlabel("% nhãn được giữ lại")
    ax1.set_ylabel("F1 Score (macro)")
    ax1.set_title("F1 Score theo % nhãn")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PR-AUC comparison
    ax2.plot(ratios_pct, curve_data["supervised_pr_auc"], "o-",
             label="Supervised-only", color="#3498db", linewidth=2, markersize=8)
    ax2.plot(ratios_pct, curve_data["semi_pr_auc"], "s-",
             label="Self-Training", color="#e74c3c", linewidth=2, markersize=8)
    ax2.set_xlabel("% nhãn được giữ lại")
    ax2.set_ylabel("PR-AUC")
    ax2.set_title("PR-AUC theo % nhãn")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Learning Curve: Supervised vs Semi-supervised",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "learning_curve_semi.png", output_dir)


# ==================================================================
# 5. Association Rules / Lift Comparison
# ==================================================================

def plot_lift_comparison(lift_df, output_dir=DEFAULT_OUTPUT):
    """
    So sánh lift giữa luật dẫn đến nghỉ việc vs ở lại.

    Parameters
    ----------
    lift_df : pd.DataFrame
        Columns: ["Rule", "Lift_Leave", "Lift_Stay"]
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    top = lift_df.head(10)
    x = np.arange(len(top))
    width = 0.35

    ax.barh(x - width / 2, top["Lift_Leave"], width,
            label="Nghỉ việc (Leave)", color="#e74c3c", edgecolor="black")
    ax.barh(x + width / 2, top["Lift_Stay"], width,
            label="Ở lại (Stay)", color="#2ecc71", edgecolor="black")

    ax.set_yticks(x)
    ax.set_yticklabels(top["Rule"].values, fontsize=10)
    ax.set_xlabel("Lift")
    ax.set_title("So sánh Lift: Nghỉ việc vs Ở lại (Top 10 luật)",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return _save(fig, "lift_comparison.png", output_dir)


def plot_clustering_comparison(comparison_data, output_dir=DEFAULT_OUTPUT):
    """So sánh các phương pháp clustering."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(comparison_data.keys())
    scores = [comparison_data[m]["silhouette"] for m in methods]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    bars = ax.bar(methods, scores, color=colors[:len(methods)], edgecolor="black")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("So sánh phương pháp Clustering", fontsize=14, fontweight="bold")

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{score:.4f}", ha="center", fontweight="bold")

    fig.tight_layout()
    return _save(fig, "clustering_comparison.png", output_dir)
