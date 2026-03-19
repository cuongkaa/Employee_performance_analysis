"""
builder.py - Feature Engineering cho HR Analytics
Tạo các đặc trưng phái sinh để cải thiện mô hình dự đoán.
"""
import pandas as pd
import numpy as np


class FeatureBuilder:
    """
    Xây dựng các đặc trưng (features) mới từ dữ liệu HR gốc.
    Các đặc trưng được thiết kế dựa trên domain knowledge về nhân sự.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ------------------------------------------------------------------
    # 1. Chỉ số hài lòng tổng hợp
    # ------------------------------------------------------------------
    def build_satisfaction_index(self) -> pd.DataFrame:
        """
        Tổng hợp các chỉ số hài lòng thành một chỉ số duy nhất.
        SatisfactionIndex = mean(JobSatisfaction, EnvironmentSatisfaction,
                                 RelationshipSatisfaction, WorkLifeBalance)
        """
        sat_cols = [
            "JobSatisfaction", "EnvironmentSatisfaction",
            "RelationshipSatisfaction", "WorkLifeBalance",
        ]
        available = [c for c in sat_cols if c in self.df.columns]
        if available:
            self.df["SatisfactionIndex"] = self.df[available].mean(axis=1)
            print(f"[FeatureBuilder] Đã tạo SatisfactionIndex từ {len(available)} cột")
        return self.df

    # ------------------------------------------------------------------
    # 2. Tốc độ thăng tiến (Career Growth)
    # ------------------------------------------------------------------
    def build_career_growth(self) -> pd.DataFrame:
        """
        CareerGrowthRate = YearsAtCompany / max(YearsSinceLastPromotion, 1)
          → Cao = được thăng chức thường xuyên
          → Thấp = bị "bỏ quên", dễ nghỉ việc

        PromotionStagnation = YearsSinceLastPromotion / max(TotalWorkingYears, 1)
          → Cao = thăng tiến chậm so với kinh nghiệm
        """
        if "YearsAtCompany" in self.df.columns and "YearsSinceLastPromotion" in self.df.columns:
            self.df["CareerGrowthRate"] = (
                self.df["YearsAtCompany"] /
                self.df["YearsSinceLastPromotion"].clip(lower=1)
            )
            print("[FeatureBuilder] Đã tạo CareerGrowthRate")

        if "YearsSinceLastPromotion" in self.df.columns and "TotalWorkingYears" in self.df.columns:
            self.df["PromotionStagnation"] = (
                self.df["YearsSinceLastPromotion"] /
                self.df["TotalWorkingYears"].clip(lower=1)
            )
            print("[FeatureBuilder] Đã tạo PromotionStagnation")

        return self.df

    # ------------------------------------------------------------------
    # 3. Chỉ số khối lượng công việc (Workload Indicator)
    # ------------------------------------------------------------------
    def build_workload_indicator(self) -> pd.DataFrame:
        """
        WorkloadScore: kết hợp OverTime + BusinessTravel + số năm không thăng chức
        → Cao = áp lực lớn, rủi ro nghỉ việc cao
        """
        score = np.zeros(len(self.df))

        # OverTime: Yes = +2
        if "OverTime" in self.df.columns:
            score += self.df["OverTime"].map({"Yes": 2, "No": 0}).fillna(0)

        # BusinessTravel: Frequently = +2, Rarely = +1
        if "BusinessTravel" in self.df.columns:
            travel_map = {
                "Travel_Frequently": 2,
                "Travel_Rarely": 1,
                "TravelRarely": 1,
                "Non-Travel": 0,
            }
            score += self.df["BusinessTravel"].map(travel_map).fillna(0)

        # Thêm điểm nếu không được thăng chức lâu (>3 năm)
        if "YearsSinceLastPromotion" in self.df.columns:
            score += (self.df["YearsSinceLastPromotion"] > 3).astype(int)

        self.df["WorkloadScore"] = score
        print("[FeatureBuilder] Đã tạo WorkloadScore")
        return self.df

    # ------------------------------------------------------------------
    # 4. Loyalty Indicator
    # ------------------------------------------------------------------
    def build_loyalty_indicator(self) -> pd.DataFrame:
        """
        LoyaltyScore = YearsAtCompany / max(TotalWorkingYears, 1)
        → Cao = nhân viên trung thành (phần lớn sự nghiệp ở công ty hiện tại)
        → Thấp = hay nhảy việc

        AvgYearsPerCompany = TotalWorkingYears / max(NumCompaniesWorked, 1)
        """
        if "YearsAtCompany" in self.df.columns and "TotalWorkingYears" in self.df.columns:
            self.df["LoyaltyScore"] = (
                self.df["YearsAtCompany"] /
                self.df["TotalWorkingYears"].clip(lower=1)
            )
            print("[FeatureBuilder] Đã tạo LoyaltyScore")

        if "TotalWorkingYears" in self.df.columns and "NumCompaniesWorked" in self.df.columns:
            self.df["AvgYearsPerCompany"] = (
                self.df["TotalWorkingYears"] /
                self.df["NumCompaniesWorked"].clip(lower=1)
            )
            print("[FeatureBuilder] Đã tạo AvgYearsPerCompany")

        return self.df

    # ------------------------------------------------------------------
    # 5. Income-to-Level Ratio
    # ------------------------------------------------------------------
    def build_income_ratio(self) -> pd.DataFrame:
        """
        IncomePerLevel = MonthlyIncome / JobLevel
        → Thấp so với level = có thể bất mãn lương
        """
        if "MonthlyIncome" in self.df.columns and "JobLevel" in self.df.columns:
            self.df["IncomePerLevel"] = (
                self.df["MonthlyIncome"] / self.df["JobLevel"].clip(lower=1)
            )
            print("[FeatureBuilder] Đã tạo IncomePerLevel")
        return self.df

    # ------------------------------------------------------------------
    # Tổng hợp: build tất cả features
    # ------------------------------------------------------------------
    def build_all(self) -> pd.DataFrame:
        """Gọi tất cả các phương thức xây dựng đặc trưng."""
        self.build_satisfaction_index()
        self.build_career_growth()
        self.build_workload_indicator()
        self.build_loyalty_indicator()
        self.build_income_ratio()

        new_features = [
            "SatisfactionIndex", "CareerGrowthRate", "PromotionStagnation",
            "WorkloadScore", "LoyaltyScore", "AvgYearsPerCompany", "IncomePerLevel",
        ]
        existing = [f for f in new_features if f in self.df.columns]
        print(f"\n[FeatureBuilder] Tổng cộng đã tạo {len(existing)} đặc trưng mới: {existing}")
        return self.df
