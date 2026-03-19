# HR Analytics Project — Phân tích hiệu suất nhân viên

Dự án phân tích dữ liệu nhân sự (HR Analytics) sử dụng các kỹ thuật khai phá dữ liệu và học máy để dự đoán và phân tích tình trạng nghỉ việc (Attrition) của nhân viên.

## 📊 Tổng quan

| Kỹ thuật | Mục tiêu | Thuật toán |
|---|---|---|
| Luật kết hợp | Tìm yếu tố dẫn đến nghỉ việc, so sánh lift Stay vs Leave | Apriori |
| Phân cụm | Profiling nhóm nhân viên + HR Strategy | K-Means, DBSCAN, HAC |
| Phân lớp có giám sát | Dự đoán nghỉ việc | Random Forest, XGBoost |
| Bán giám sát | Dự đoán khi thiếu nhãn (5/10/20/30%) | Self-Training |
| Hồi quy | Dự đoán mức hài lòng (JobSatisfaction) | Linear, RF, GBR |
| Đánh giá | Metrics cho dữ liệu mất cân bằng | PR-AUC, F1, Silhouette, SHAP, MAE/RMSE |

## 📁 Cấu trúc dự án

```
Employee_performance_analysis/
├── configs/params.yaml           # Tham số thí nghiệm (hyperparams, seed, split)
├── data/
│   ├── raw/HR_Analytics.csv      # Dữ liệu gốc (Kaggle)
│   └── processed/                # Dữ liệu sau làm sạch
├── docs/
│   └── data_dictionary.md        # Mô tả chi tiết 38 cột dữ liệu
├── notebooks/
│   ├── 01_eda.ipynb              # Khám phá dữ liệu + Insight
│   ├── 02_preprocess_feature.ipynb  # Tiền xử lý + Feature Engineering
│   ├── 03_mining_or_clustering.ipynb # Luật kết hợp + Phân cụm
│   ├── 04_modeling.ipynb         # Phân lớp (RF, XGB) + SHAP
│   ├── 04b_semi_supervised.ipynb # Bán giám sát + Learning Curve
│   └── 05_evaluation_report.ipynb # Tổng hợp + Khuyến nghị
├── src/
│   ├── data/
│   │   ├── loader.py             # Đọc dữ liệu & config
│   │   └── cleaner.py            # Làm sạch, rời rạc hóa, mã hóa
│   ├── features/
│   │   └── builder.py            # Feature Engineering (7 đặc trưng mới)
│   ├── mining/
│   │   ├── association.py        # Apriori + Lift comparison + HR policy
│   │   └── clustering.py         # KMeans/DBSCAN/HAC + Profiling + HR Strategy
│   ├── models/
│   │   ├── supervised.py         # Random Forest & XGBoost
│   │   ├── semi_supervised.py    # Self-Training (multi-ratio) + Pseudo-label risk
│   │   └── regression.py         # Hồi quy + Leakage check
│   ├── evaluation/
│   │   └── metrics.py            # PR-AUC, F1, MAE/RMSE, SHAP, Comparison tables
│   └── visualization/
│       └── plots.py              # 12 hàm vẽ biểu đồ dùng chung
├── scripts/run_pipeline.py       # Chạy toàn bộ pipeline
├── outputs/
│   ├── figures/                  # Biểu đồ tự động tạo
│   ├── tables/                   # Bảng kết quả CSV
│   └── models/                   # Mô hình đã huấn luyện
├── app.py                        # Streamlit Demo App (bonus)
├── requirements.txt
└── README.md
```

## 🔬 Pipeline

```
Dữ liệu gốc (CSV)
    ↓
Tiền xử lý (làm sạch, xử lý thiếu, chuẩn hóa BusinessTravel)
    ↓
Feature Engineering (7 đặc trưng mới: SatisfactionIndex, CareerGrowth, ...)
    ↓
┌─────────────────────┬──────────────────────┬─────────────────────┐
│  Luật kết hợp       │  Phân cụm            │  Mô hình dự đoán    │
│  (Apriori)          │  (KMeans/DBSCAN/HAC) │  (RF, XGBoost)      │
│  • Lift comparison  │  • Profiling         │  • PR-AUC, F1       │
│  • HR Policy        │  • HR Strategy       │  • SHAP explain     │
└─────────────────────┴──────────────────────┴─────────────────────┘
    ↓                      ↓                       ↓
Bán giám sát (5/10/20/30% labels)     Hồi quy (JobSatisfaction)
    • Learning Curve                       • MAE/RMSE, R²
    • Pseudo-label Risk                    • Leakage Check
    ↓
Tổng hợp: Bảng + Biểu đồ so sánh + Khuyến nghị HR
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Tải dataset **IBM HR Analytics Employee Attrition & Performance** từ [Kaggle](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset) và đặt file `HR_Analytics.csv` vào thư mục `data/raw/`.

### 3. Chạy pipeline

```bash
python scripts/run_pipeline.py
```

Pipeline thực hiện tuần tự 11 bước:
1. Đọc & cấu hình
2. Làm sạch dữ liệu
3. Feature Engineering (7 đặc trưng mới)
4. EDA & Biểu đồ
5. Tìm luật kết hợp (Apriori) + so sánh Lift + gợi ý chính sách
6. Phân cụm (KMeans, DBSCAN, HAC) + Profiling + HR Strategy
7. Huấn luyện Random Forest & XGBoost
8. Bán giám sát (5/10/20/30% nhãn) + Pseudo-label risk
9. Tổng hợp so sánh (bảng + biểu đồ)
10. Giải thích SHAP (top 10 features)
11. Hồi quy mức hài lòng + kiểm tra leakage

### 4. Notebook

```bash
jupyter notebook notebooks/
```

### 5. Demo App (Bonus)

```bash
streamlit run app.py
```

## ⚙️ Thiết lập thí nghiệm (Experimental Setup)

Tất cả tham số được ghi trong `configs/params.yaml`:

```yaml
data:
  test_size: 0.2          # 80/20 split
model:
  random_state: 42        # Reproducibility seed
  rf_n_estimators: 100    # Random Forest: 100 cây
  xgb_n_estimators: 100   # XGBoost: 100 rounds
  xgb_learning_rate: 0.1  # XGBoost: lr = 0.1
  xgb_max_depth: 5        # XGBoost: depth = 5
semi_supervised:
  label_ratios: [0.05, 0.10, 0.20, 0.30]
```

## 📈 Dataset

- **Nguồn**: IBM HR Analytics (Kaggle)
- **Kích thước**: 1,480 dòng × 38 thuộc tính (1,470 nhân viên duy nhất)
- **Target**: `Attrition` (Yes/No) — tỷ lệ mất cân bằng (~16% Yes)
- **Chi tiết**: Xem `docs/data_dictionary.md` cho mô tả đầy đủ

## ♻️ Reproducibility

```bash
# 1. Cài đặt
pip install -r requirements.txt

# 2. Cập nhật đường dẫn dữ liệu (nếu cần)
# Sửa configs/params.yaml → data.raw_path

# 3. Chạy pipeline
python scripts/run_pipeline.py

# 4. Kết quả
# outputs/figures/  → biểu đồ
# outputs/tables/   → bảng CSV
# → Output phải giống report
```
