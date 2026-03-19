"""
cleaner.py - Làm sạch, mã hóa và rời rạc hóa dữ liệu HR
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class HRDataCleaner:
    """
    Lớp xử lý dữ liệu HR Analytics:
      - Xử lý giá trị thiếu
      - Loại bỏ cột không cần thiết
      - Mã hóa biến phân loại
      - Rời rạc hóa biến liên tục cho Association Rules
    """

    COLS_TO_DROP = [
        "EmpID", "EmployeeCount", "EmployeeNumber",
        "Over18", "StandardHours",
    ]

    def __init__(self, target_col: str = "Attrition"):
        self.target_col = target_col
        self.label_encoders: dict[str, LabelEncoder] = {}

    # ------------------------------------------------------------------
    # Bước 1: Làm sạch cơ bản
    # ------------------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý thiếu, loại cột dư, chuẩn hóa kiểu dữ liệu."""
        df = df.copy()

        # Xử lý giá trị thiếu bằng median (cột số) hoặc mode (cột chuỗi)
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if df[col].dtype in ("float64", "int64"):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

        # Chuẩn hóa BusinessTravel (fix lỗi nhập liệu TravelRarely → Travel_Rarely)
        if "BusinessTravel" in df.columns:
            df["BusinessTravel"] = df["BusinessTravel"].replace("TravelRarely", "Travel_Rarely")

        # Loại bỏ cột không mang thông tin
        df = df.drop(columns=[c for c in self.COLS_TO_DROP if c in df.columns])

        # Loại bỏ cột AgeGroup (trùng thông tin với Age)
        if "AgeGroup" in df.columns:
            df = df.drop(columns=["AgeGroup"])

        # Loại bỏ cột SalarySlab (trùng thông tin với MonthlyIncome)
        if "SalarySlab" in df.columns:
            df = df.drop(columns=["SalarySlab"])

        print(f"[Cleaner] Sau khi làm sạch: {df.shape[0]} dòng, {df.shape[1]} cột")
        return df

    # ------------------------------------------------------------------
    # Bước 2: Mã hóa biến phân loại → số (cho mô hình ML)
    # ------------------------------------------------------------------
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mã hóa biến phân loại bằng one-hot encoding (get_dummies).
        Cột target (Attrition) được chuyển thành 0/1.
        """
        df = df.copy()

        # Mã hóa target
        if self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].map({"Yes": 1, "No": 0})

        # One-hot encode các cột phân loại còn lại
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        return df

    # ------------------------------------------------------------------
    # Bước 3: Rời rạc hóa cho Association Rule Mining
    # ------------------------------------------------------------------
    def discretize_for_mining(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển đổi các biến số thành biến định danh (bins)
        để phục vụ thuật toán Apriori tìm luật kết hợp.
        Chỉ giữ lại các cột phân loại quan trọng.
        """
        df_disc = df.copy()

        # Thu nhập hàng tháng → 3 mức
        if "MonthlyIncome" in df_disc.columns:
            df_disc["MonthlyIncome"] = pd.qcut(
                df["MonthlyIncome"], q=3,
                labels=["Low_Income", "Mid_Income", "High_Income"],
            )

        # Tuổi → 3 nhóm
        if "Age" in df_disc.columns:
            df_disc["Age"] = pd.cut(
                df["Age"], bins=[0, 30, 45, 100],
                labels=["Young", "Middle-Aged", "Senior"],
            )

        # Số năm kinh nghiệm → 3 nhóm
        if "TotalWorkingYears" in df_disc.columns:
            df_disc["TotalWorkingYears"] = pd.cut(
                df["TotalWorkingYears"], bins=[-1, 5, 15, 50],
                labels=["Junior", "Mid-Level", "Senior-Level"],
            )

        # Khoảng cách từ nhà → 3 mức
        if "DistanceFromHome" in df_disc.columns:
            df_disc["DistanceFromHome"] = pd.cut(
                df["DistanceFromHome"], bins=[0, 5, 15, 30],
                labels=["Near", "Medium", "Far"],
            )

        # Sự hài lòng công việc → 2 nhóm
        if "JobSatisfaction" in df_disc.columns:
            df_disc["JobSatisfaction"] = df_disc["JobSatisfaction"].map(
                {1: "Low_Satisfaction", 2: "Low_Satisfaction",
                 3: "High_Satisfaction", 4: "High_Satisfaction"}
            )

        # Overtime giữ nguyên (đã là Yes/No)

        # Số năm ở công ty
        if "YearsAtCompany" in df_disc.columns:
            df_disc["YearsAtCompany"] = pd.cut(
                df["YearsAtCompany"], bins=[-1, 3, 10, 50],
                labels=["Short_Tenure", "Mid_Tenure", "Long_Tenure"],
            )

        # Chỉ giữ lại các cột định tính quan trọng cho Apriori
        mining_cols = [
            "Attrition", "Age", "MonthlyIncome", "OverTime",
            "TotalWorkingYears", "DistanceFromHome", "JobSatisfaction",
            "YearsAtCompany", "MaritalStatus", "Department", "JobRole",
            "EnvironmentSatisfaction", "WorkLifeBalance",
        ]
        mining_cols = [c for c in mining_cols if c in df_disc.columns]

        # Chuyển các cột số còn lại sang categorical (nếu có)
        for col in mining_cols:
            if df_disc[col].dtype in ("int64", "float64"):
                df_disc[col] = df_disc[col].astype(str)

        df_disc = df_disc[mining_cols]

        print(f"[Cleaner] Đã rời rạc hóa và chọn {len(mining_cols)} cột cho mining")
        return df_disc

    # ------------------------------------------------------------------
    # Tiện ích: lưu dữ liệu đã xử lý
    # ------------------------------------------------------------------
    def save_processed(self, df: pd.DataFrame, path: str = "data/processed/hr_cleaned.csv"):
        """Lưu dữ liệu đã làm sạch ra file CSV."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[Cleaner] Đã lưu dữ liệu tại: {path}")