"""
regression.py - Hồi quy mức độ hài lòng / điểm hiệu suất
Kiểm tra data leakage và đánh giá MAE/RMSE.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def check_leakage(df, target_col, threshold=0.85):
    """
    Kiểm tra data leakage: tìm các cột có tương quan quá cao với target.

    Parameters
    ----------
    df : pd.DataFrame
        Dữ liệu đã được mã hóa (toàn số).
    target_col : str
        Cột mục tiêu (VD: 'JobSatisfaction').
    threshold : float
        Ngưỡng tương quan để cảnh báo leakage.

    Returns
    -------
    dict  {"leaked_features": [...], "correlations": pd.Series}
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        print(f"[Leakage] Cột '{target_col}' không phải kiểu số.")
        return {"leaked_features": [], "correlations": pd.Series()}

    corr = numeric_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    leaked = corr[corr >= threshold].index.tolist()

    if leaked:
        print(f"[Leakage] ⚠️ CẢNH BÁO: {len(leaked)} cột có tương quan >= {threshold} với '{target_col}':")
        for feat in leaked:
            print(f"  - {feat}: corr = {corr[feat]:.4f}")
    else:
        print(f"[Leakage] ✅ Không phát hiện data leakage (threshold={threshold})")

    return {"leaked_features": leaked, "correlations": corr.head(15)}


def train_satisfaction_regressor(
    df,
    target_col="JobSatisfaction",
    test_size=0.2,
    random_state=42,
    drop_cols=None,
):
    """
    Huấn luyện mô hình hồi quy dự đoán mức độ hài lòng.

    Parameters
    ----------
    df : pd.DataFrame
        Dữ liệu đã mã hóa.
    target_col : str
        Cột mục tiêu hồi quy.
    test_size : float
    random_state : int
    drop_cols : list, optional
        Các cột cần loại bỏ (VD: cột bị leakage).

    Returns
    -------
    dict  {
        "models": {"Linear": ..., "RF": ..., "GBR": ...},
        "results": pd.DataFrame (bảng so sánh),
        "best_model_name": str,
        "X_test": ..., "y_test": ...,
    }
    """
    # Chuẩn bị dữ liệu
    df_reg = df.select_dtypes(include=[np.number]).copy()

    # Loại bỏ cột target classification (Attrition) và cột leak
    cols_drop = ["Attrition"]
    if drop_cols:
        cols_drop.extend(drop_cols)

    # Tự động loại bỏ các cột phái sinh có chứa target
    # (VD: SatisfactionIndex chứa JobSatisfaction → leakage)
    derived_leak_cols = ["SatisfactionIndex"]
    if target_col in ("JobSatisfaction", "EnvironmentSatisfaction",
                       "RelationshipSatisfaction", "WorkLifeBalance"):
        cols_drop.extend(derived_leak_cols)
        print(f"[Regression] ⚠️ Loại bỏ {derived_leak_cols} (phái sinh chứa {target_col})")

    cols_drop = [c for c in cols_drop if c in df_reg.columns]

    if target_col not in df_reg.columns:
        print(f"[Regression] Cột '{target_col}' không tìm thấy.")
        return None

    X = df_reg.drop(columns=cols_drop + [target_col], errors="ignore")
    y = df_reg[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    # Chuẩn hóa
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện 3 mô hình
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=random_state
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=random_state
        ),
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R²": round(r2, 4),
        })
        fitted_models[name] = model

        print(f"[Regression] {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    results_df = pd.DataFrame(results)
    best_name = results_df.loc[results_df["MAE"].idxmin(), "Model"]
    print(f"\n[Regression] Mô hình tốt nhất (MAE thấp nhất): {best_name}")

    return {
        "models": fitted_models,
        "results": results_df,
        "best_model_name": best_name,
        "X_test": pd.DataFrame(X_test_scaled, columns=X.columns),
        "y_test": y_test,
        "feature_names": X.columns.tolist(),
    }
