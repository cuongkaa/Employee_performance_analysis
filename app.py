"""
app.py - Streamlit Demo App cho HR Analytics
Chạy: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(__file__))

from src.data.loader import load_data, load_config
from src.data.cleaner import HRDataCleaner
from src.features.builder import FeatureBuilder

# ============================================================
# Cấu hình trang
# ============================================================
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_and_prepare_data():
    """Tải và chuẩn bị dữ liệu."""
    config = load_config("configs/params.yaml")
    df = pd.read_csv(config["data"]["raw_path"])
    cleaner = HRDataCleaner(target_col="Attrition")
    df_clean = cleaner.clean(df)
    fb = FeatureBuilder(df_clean)
    df_featured = fb.build_all()
    return df, df_clean, df_featured, config


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("📊 HR Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Chọn trang",
    ["🏠 Tổng quan & EDA", "🔍 Luật kết hợp", "📦 Phân cụm", "🤖 Dự đoán nghỉ việc"],
)

# Tải dữ liệu
try:
    df_raw, df_clean, df_featured, config = load_and_prepare_data()
except Exception as e:
    st.error(f"Lỗi khi tải dữ liệu: {e}")
    st.info("Đảm bảo file `data/raw/HR_Analytics.csv` tồn tại.")
    st.stop()


# ============================================================
# Trang 1: Tổng quan & EDA
# ============================================================
if page == "🏠 Tổng quan & EDA":
    st.title("🏠 Tổng quan & Khám phá dữ liệu")
    st.markdown("---")

    # Metrics tổng quan
    col1, col2, col3, col4 = st.columns(4)
    total_emp = len(df_clean)
    attrition_yes = (df_clean["Attrition"] == "Yes").sum()
    attrition_rate = attrition_yes / total_emp * 100

    col1.metric("Tổng nhân viên", f"{total_emp:,}")
    col2.metric("Nghỉ việc", f"{attrition_yes}")
    col3.metric("Tỷ lệ nghỉ việc", f"{attrition_rate:.1f}%")
    col4.metric("Số features", f"{df_clean.shape[1]}")

    st.markdown("---")

    # Phân bố Attrition
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df_clean, names="Attrition", title="Phân bố Attrition",
                     color_discrete_map={"Yes": "#ff4b4b", "No": "#00cc96"},
                     hole=0.4)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        factor = st.selectbox(
            "Chọn yếu tố phân tích",
            ["OverTime", "Department", "JobRole", "MaritalStatus",
             "BusinessTravel", "Gender", "EducationField"],
        )
        ct = pd.crosstab(df_clean[factor], df_clean["Attrition"], normalize="index") * 100
        fig = px.bar(ct, barmode="group", title=f"Tỷ lệ Attrition theo {factor}",
                     color_discrete_map={"Yes": "#ff4b4b", "No": "#00cc96"})
        fig.update_layout(
            yaxis_title="Tỷ lệ (%)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Distributions
    st.subheader("📈 Phân bố biến số")
    num_col = st.selectbox(
        "Chọn biến số",
        ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
         "DistanceFromHome", "JobSatisfaction"],
    )
    fig = px.histogram(df_clean, x=num_col, color="Attrition", barmode="overlay",
                       color_discrete_map={"Yes": "#ff4b4b", "No": "#00cc96"},
                       opacity=0.8,
                       title=f"Phân bố {num_col} theo Attrition")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("🔥 Ma trận tương quan")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    corr = df_clean[numeric_cols].corr()
    # Filter to display top 15 most correlated features with Attrition? Or keep all but readable
    # To keep all numeric cols, a diverging colorscale like RdBu_r from -1 to 1 is best
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    title="Ma trận tương quan (Đỏ: Thuận, Xanh: Nghịch)")
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(
        height=700,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 20, "r": 20, "t": 50, "b": 100}
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Trang 2: Luật kết hợp
# ============================================================
elif page == "🔍 Luật kết hợp":
    st.title("🔍 Luật kết hợp (Association Rules)")
    st.markdown("---")

    from src.mining.association import get_attrition_rules, compare_lift_stay_vs_leave

    min_support = st.sidebar.slider("Min Support", 0.01, 0.20, 0.05, 0.01)
    min_lift = st.sidebar.slider("Min Lift", 1.0, 3.0, 1.2, 0.1)

    cleaner = HRDataCleaner(target_col="Attrition")
    df_disc = cleaner.discretize_for_mining(df_clean)

    with st.spinner("Đang tìm luật kết hợp..."):
        rules = get_attrition_rules(df_disc, min_support=min_support, min_threshold=min_lift)

    if not rules.empty:
        st.success(f"Tìm thấy {len(rules)} luật dẫn đến nghỉ việc")

        # Hiển thị top rules
        display_df = rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(15)
        display_df["antecedents"] = display_df["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        display_df["consequents"] = display_df["consequents"].apply(lambda x: ", ".join(sorted(x)))
        st.dataframe(display_df, use_container_width=True)

        # Lift chart
        fig = px.bar(display_df.head(10), x="lift", y="antecedents", orientation="h",
                     title="Top 10 luật theo Lift", color="lift",
                     color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Không tìm thấy luật. Thử giảm Min Support hoặc Min Lift.")

    # So sánh Lift
    st.subheader("⚖️ So sánh Lift: Nghỉ việc vs Ở lại")
    with st.spinner("Đang so sánh..."):
        lift_comp = compare_lift_stay_vs_leave(df_disc, min_support=min_support)

    if not lift_comp.empty:
        st.dataframe(lift_comp.head(15), use_container_width=True)


# ============================================================
# Trang 3: Phân cụm
# ============================================================
elif page == "📦 Phân cụm":
    st.title("📦 Phân cụm nhân viên (Clustering)")
    st.markdown("---")

    from src.mining.clustering import run_kmeans, profile_clusters

    n_clusters = st.sidebar.slider("Số cụm (K)", 2, 8, 3)
    cluster_features = config["mining"]["cluster_features"]
    cluster_features = [f for f in cluster_features if f in df_clean.columns]

    with st.spinner("Đang phân cụm..."):
        df_clustered, _, sil_score = run_kmeans(df_clean, cluster_features, n_clusters=n_clusters)
        profile = profile_clusters(df_clustered, cluster_features)

    st.metric("Silhouette Score", f"{sil_score:.4f}")

    # Profile table
    st.subheader("📋 Profile các cụm")
    st.dataframe(profile, use_container_width=True)

    # Scatter plot
    st.subheader("📊 Biểu đồ phân cụm")
    x_col = st.selectbox("Trục X", cluster_features, index=0)
    y_col = st.selectbox("Trục Y", cluster_features, index=1)

    fig = px.scatter(df_clustered, x=x_col, y=y_col, color="Cluster",
                     title=f"Phân cụm: {x_col} vs {y_col}",
                     color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

    # Cluster distribution
    cluster_counts = df_clustered["Cluster"].value_counts().sort_index()
    fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                 labels={"x": "Cụm", "y": "Số lượng"},
                 title="Phân bố số lượng nhân viên theo cụm")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Trang 4: Dự đoán nghỉ việc
# ============================================================
elif page == "🤖 Dự đoán nghỉ việc":
    st.title("🤖 Dự đoán nghỉ việc")
    st.markdown("---")

    st.info("Nhập thông tin nhân viên để dự đoán khả năng nghỉ việc.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Tuổi", 18, 60, 30)
        monthly_income = st.number_input("Thu nhập hàng tháng ($)", 100, 20000, 5000)
        distance = st.number_input("Khoảng cách từ nhà (km)", 1, 30, 5)
        stock_option = st.selectbox("Quyền chọn cổ phiếu (Stock Option)", [0, 1, 2, 3])

    with col2:
        job_satisfaction = st.slider("Hài lòng công việc", 1, 4, 3)
        env_satisfaction = st.slider("Hài lòng môi trường", 1, 4, 3)
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)

    with col3:
        overtime = st.selectbox("Làm thêm giờ", ["No", "Yes"])
        years_at_company = st.number_input("Số năm tại công ty", 0, 40, 5)
        total_working_years = st.number_input("Tổng kinh nghiệm (năm)", 0, 40, 10)
        marital_status = st.selectbox("Tình trạng hôn nhân", ["Single", "Married", "Divorced"])

    if st.button("🔮 Dự đoán", type="primary"):
        try:
            from src.models.supervised import train_random_forest
            from src.data.cleaner import HRDataCleaner

            cleaner = HRDataCleaner(target_col="Attrition")
            df_enc = cleaner.encode(df_clean.copy())

            X = df_enc.drop(columns=["Attrition"])
            y = df_enc["Attrition"]

            model = train_random_forest(X, y, n_estimators=100, random_state=42)

            # Tạo 1 dòng dữ liệu mẫu (dùng mode làm nền cho các cột không nhập)
            sample_data = df_clean.drop(columns=["Attrition"], errors="ignore").mode().iloc[0:1].copy()
            
            # Ghi đè các giá trị người dùng nhập
            sample_data["Age"] = age
            sample_data["MonthlyIncome"] = monthly_income
            sample_data["DistanceFromHome"] = distance
            sample_data["JobSatisfaction"] = job_satisfaction
            sample_data["EnvironmentSatisfaction"] = env_satisfaction
            sample_data["WorkLifeBalance"] = work_life_balance
            sample_data["YearsAtCompany"] = years_at_company
            sample_data["TotalWorkingYears"] = total_working_years
            sample_data["OverTime"] = overtime
            sample_data["MaritalStatus"] = marital_status
            sample_data["StockOptionLevel"] = stock_option
            
            # Tạo feature phái sinh (WorkloadScore, SatisfactionIndex...)
            fb_custom = FeatureBuilder(sample_data)
            sample_featured = fb_custom.build_all()
            
            # Encode
            sample_enc = cleaner.encode(sample_featured)
            
            # Khớp cột (điền 0 cho các phân loại one-hot bị thiếu)
            input_data = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
            for col in sample_enc.columns:
                if col in input_data.columns:
                    input_data[col] = sample_enc[col].values[0]

            prob_ml = model.predict_proba(input_data)[0][1]
            
            # THUẬT TOÁN ĐIỀU CHỈNH THEO TRỌNG SỐ TÙY CHỈNH (EXPERT SYSTEM)
            # Dựa trên phân cấp độ ưu tiên do người dùng chỉ định:
            
            # 1. Salary (Max 40% rủi ro)
            if monthly_income <= 1000: sal_score = 0.40
            elif monthly_income <= 3000: sal_score = 0.25
            elif monthly_income <= 5000: sal_score = 0.10
            else: sal_score = 0.0

            # 2. Satisfaction (Max ~25% rủi ro)
            sat_score = (4 - job_satisfaction) * 0.042 + (4 - env_satisfaction) * 0.041
            
            # 3. WLB + Overtime (Max 15% rủi ro)
            ot_score = 0.10 if overtime == "Yes" else 0.0
            wlb_score = (4 - work_life_balance) * 0.016
            
            # 4. Experience & Tenure (Max 10% rủi ro)
            if years_at_company <= 1: exp_score = 0.10
            elif years_at_company <= 3: exp_score = 0.05
            elif years_at_company >= 8: exp_score = -0.10  # Ổn định
            else: exp_score = 0.0
                
            # 5. Others (Max 10% rủi ro)
            dist_score = 0.05 if distance >= 20 else 0.0
            mar_score = 0.05 if marital_status == "Single" else 0.0
            
            # Tổng hợp rủi ro
            business_risk = sal_score + sat_score + ot_score + wlb_score + exp_score + dist_score + mar_score
            business_risk = max(0.01, min(0.98, business_risk))
            
            # Kết hợp ML và Hệ chuyên gia (Chống Out-of-Distribution)
            prob = max(prob_ml, business_risk)

            pred = "Có khả năng nghỉ việc" if prob >= 0.5 else "Có khả năng ở lại"

            st.markdown("---")
            col1, col2 = st.columns(2)
            col1.metric("Kết quả", pred)
            col2.metric("Xác suất nghỉ việc", f"{prob:.1%}")

            # --- TẠO ĐỀ XUẤT ĐỘNG (DỰA TRÊN CHỈ SỐ RỦI RO LỚN NHẤT) ---
            suggestions = []
            if sal_score > 0:
                suggestions.append("- 💰 **Lương:** Cần xem xét tăng lương cơ bản hoặc bổ sung phụ cấp.")
            if sat_score >= 0.08:
                suggestions.append("- 🗣️ **Tâm lý:** Tổ chức họp 1-1 khẩn cấp để gỡ rối các bức xúc về môi trường/công việc.")
            if ot_score > 0 or wlb_score > 0:
                suggestions.append("- ⚖️ **Áp lực:** Giảm tải Overtime, phân bổ lại Task để cân bằng Work-Life Balance.")
            if dist_score > 0:
                suggestions.append("- 🚗 **Di chuyển:** Hỗ trợ chi phí đi lại hoặc linh hoạt cho phép làm việc Remote/Hybrid.")
            if exp_score > 0:
                suggestions.append("- 🌱 **Định hướng:** Nhân sự mới vỡ mộng, cần có Mentor kèm cặp kĩ hơn.")
            if stock_option == 0:
                suggestions.append("- 📈 **Ràng buộc:** Đề xuất cấp Stock Option (ESOP) để tạo cam kết gắn bó dài hạn.")

            if not suggestions:
                suggestions.append("- Duy trì tương tác thường xuyên để nắm bắt các lý do ẩn khác.")
            
            suggestions_md = "\n".join(suggestions)

            if prob >= 0.5:
                st.error("⚠️ RỦI RO CAO — Cần can thiệp ngay!")
                st.markdown(f"**Đề xuất hành động khẩn cấp:**\n{suggestions_md}")
            elif prob >= 0.25:
                st.warning("⚡ RỦI RO TRUNG BÌNH — Cần theo dõi")
                st.markdown(f"**Khuyến nghị phòng ngừa:**\n{suggestions_md}")
            else:
                st.success("✅ ỔN ĐỊNH — Tiếp tục duy trì chính sách")

        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")


# ============================================================
# Footer
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("**HR Analytics Project**")
st.sidebar.markdown("Data Mining — Final Project")
st.sidebar.markdown("Dataset: IBM HR Analytics (Kaggle)")
