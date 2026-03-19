"""
association.py - Thuật toán Apriori tìm luật kết hợp (Association Rules)
Mục tiêu: tìm các yếu tố dẫn đến Attrition = Yes/No, so sánh lift, gợi ý chính sách
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def get_attrition_rules(
    df: pd.DataFrame,
    min_support: float = 0.05,
    min_threshold: float = 1.2,
    target_label: str = "Attrition_Yes",
) -> pd.DataFrame:
    """
    Tìm luật kết hợp có consequent chứa Attrition_Yes.

    Parameters
    ----------
    df : pd.DataFrame
        Dữ liệu đã được rời rạc hóa (cleaner.discretize_for_mining).
    min_support : float
        Ngưỡng support tối thiểu cho Apriori.
    min_threshold : float
        Ngưỡng lift tối thiểu cho luật.
    target_label : str
        Nhãn target trong one-hot encoding.

    Returns
    -------
    pd.DataFrame
        Bảng luật kết hợp, sắp xếp theo lift giảm dần.
    """
    # Bước 1: One-hot encoding toàn bộ dữ liệu đã rời rạc hóa
    df_onehot = pd.get_dummies(df).astype(bool)

    # Bước 2: Tìm tập phổ biến (Frequent Itemsets)
    frequent_itemsets = apriori(
        df_onehot,
        min_support=min_support,
        use_colnames=True,
    )

    if frequent_itemsets.empty:
        print("[Association] Không tìm thấy tập phổ biến. Thử giảm min_support.")
        return pd.DataFrame()

    # Bước 3: Sinh luật kết hợp
    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=min_threshold,
    )

    # Bước 4: Lọc luật có consequent chứa target (Attrition_Yes)
    target_rules = rules[
        rules["consequents"].apply(lambda x: target_label in str(x))
    ].copy()

    # Sắp xếp theo lift giảm dần
    target_rules = target_rules.sort_values("lift", ascending=False).reset_index(drop=True)

    print(f"[Association] Tìm thấy {len(target_rules)} luật dẫn đến {target_label}")
    return target_rules


def compare_lift_stay_vs_leave(
    df: pd.DataFrame,
    min_support: float = 0.05,
    min_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    So sánh lift giữa luật dẫn đến nghỉ việc (Leave) vs ở lại (Stay).

    Returns
    -------
    pd.DataFrame
        Columns: [Rule, Lift_Leave, Lift_Stay, Lift_Gap, Direction]
    """
    df_onehot = pd.get_dummies(df).astype(bool)

    frequent_itemsets = apriori(
        df_onehot, min_support=min_support, use_colnames=True,
    )
    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(
        frequent_itemsets, metric="lift", min_threshold=min_threshold,
    )

    # Luật dẫn đến nghỉ việc
    leave_rules = rules[
        rules["consequents"].apply(lambda x: "Attrition_Yes" in str(x))
    ].copy()
    leave_rules["antecedent_str"] = leave_rules["antecedents"].apply(
        lambda x: ", ".join(sorted(x))
    )

    # Luật dẫn đến ở lại
    stay_rules = rules[
        rules["consequents"].apply(lambda x: "Attrition_No" in str(x))
    ].copy()
    stay_rules["antecedent_str"] = stay_rules["antecedents"].apply(
        lambda x: ", ".join(sorted(x))
    )

    # Merge để so sánh
    leave_lift = leave_rules.groupby("antecedent_str")["lift"].max().reset_index()
    leave_lift.columns = ["Rule", "Lift_Leave"]

    stay_lift = stay_rules.groupby("antecedent_str")["lift"].max().reset_index()
    stay_lift.columns = ["Rule", "Lift_Stay"]

    comparison = pd.merge(leave_lift, stay_lift, on="Rule", how="outer").fillna(1.0)
    comparison["Lift_Gap"] = comparison["Lift_Leave"] - comparison["Lift_Stay"]
    comparison["Direction"] = comparison["Lift_Gap"].apply(
        lambda x: "→ Nghỉ việc" if x > 0.1 else ("→ Ở lại" if x < -0.1 else "Trung tính")
    )
    comparison = comparison.sort_values("Lift_Gap", ascending=False).reset_index(drop=True)

    print(f"[Association] So sánh lift: {len(comparison)} tổ hợp yếu tố")
    print(f"  - Thiên về nghỉ việc: {(comparison['Direction'] == '→ Nghỉ việc').sum()}")
    print(f"  - Thiên về ở lại: {(comparison['Direction'] == '→ Ở lại').sum()}")
    return comparison


def suggest_hr_policies(leave_rules: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Phân tích top luật kết hợp → đề xuất chính sách HR cụ thể.

    Parameters
    ----------
    leave_rules : pd.DataFrame
        Luật kết hợp dẫn đến nghỉ việc (output của get_attrition_rules).
    top_n : int

    Returns
    -------
    list[dict]  Mỗi dict: {"rule": ..., "lift": ..., "insight": ..., "policy": ...}
    """
    # Mapping từ yếu tố → insight + chính sách
    policy_map = {
        "OverTime_Yes": {
            "insight": "Nhân viên làm thêm giờ có tỷ lệ nghỉ việc cao",
            "policy": "Giảm OT bắt buộc, thưởng OT công bằng, theo dõi work-life balance",
        },
        "Travel_Frequently": {
            "insight": "Đi công tác thường xuyên gây kiệt sức",
            "policy": "Áp dụng chính sách hybrid/remote, giảm tần suất đi công tác",
        },
        "Low_Satisfaction": {
            "insight": "Mức hài lòng công việc thấp dẫn đến nghỉ việc",
            "policy": "Khảo sát định kỳ, cải thiện môi trường, tăng cơ hội phát triển",
        },
        "Low_Income": {
            "insight": "Thu nhập thấp so với thị trường",
            "policy": "Điều chỉnh lương theo thị trường, thưởng hiệu suất, cổ phiếu",
        },
        "Single": {
            "insight": "Nhân viên độc thân có xu hướng nghỉ việc cao hơn",
            "policy": "Xây dựng team bonding, mentor program, career path rõ ràng",
        },
        "Short_Tenure": {
            "insight": "Nhân viên mới (< 3 năm) dễ nghỉ việc",
            "policy": "Chương trình onboarding tốt hơn, mentor trong 2 năm đầu",
        },
        "Young": {
            "insight": "Nhân viên trẻ (< 30) có tỷ lệ nghỉ việc cao",
            "policy": "Career path rõ ràng, đào tạo kỹ năng, tăng engagement",
        },
        "Far": {
            "insight": "Khoảng cách xa từ nhà đến công ty",
            "policy": "Hỗ trợ đi lại, cho phép WFH, flexible schedule",
        },
        "Junior": {
            "insight": "Nhân viên ít kinh nghiệm (< 5 năm) dễ chuyển việc",
            "policy": "Đào tạo chuyên sâu, lộ trình thăng tiến rõ ràng",
        },
    }

    if leave_rules.empty:
        print("[Policy] Không có luật kết hợp để phân tích.")
        return []

    suggestions = []
    for idx, row in leave_rules.head(top_n).iterrows():
        antecedent_str = ", ".join(sorted(row["antecedents"]))
        lift_val = row["lift"]

        # Tìm insight và chính sách phù hợp
        matched_insight = "Tổ hợp yếu tố này có lift cao → rủi ro nghỉ việc"
        matched_policy = "Cần xem xét kỹ tổ hợp yếu tố và can thiệp sớm"

        for key, value in policy_map.items():
            if key in antecedent_str:
                matched_insight = value["insight"]
                matched_policy = value["policy"]
                break

        suggestion = {
            "rule": antecedent_str,
            "lift": round(lift_val, 3),
            "confidence": round(row.get("confidence", 0), 3),
            "insight": matched_insight,
            "policy": matched_policy,
        }
        suggestions.append(suggestion)

    # In ra suggestions
    print(f"\n{'='*60}")
    print("  ĐỀ XUẤT CHÍNH SÁCH HR (dựa trên luật kết hợp)")
    print(f"{'='*60}")
    for i, s in enumerate(suggestions, 1):
        print(f"\n  {i}. Luật: {s['rule']}")
        print(f"     Lift: {s['lift']} | Confidence: {s['confidence']}")
        print(f"     📊 Insight: {s['insight']}")
        print(f"     ✅ Chính sách: {s['policy']}")

    return suggestions