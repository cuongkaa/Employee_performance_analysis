# Data Dictionary — IBM HR Analytics Employee Attrition & Performance

## Nguồn dữ liệu
- **Dataset**: IBM HR Analytics Employee Attrition & Performance
- **Link**: [Kaggle](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Kích thước**: 1,480 dòng × 38 cột (1,470 nhân viên duy nhất)
- **Loại dữ liệu**: Dạng bảng (tabular), dữ liệu giả lập (synthetic)

## Biến mục tiêu (Target)

| Cột | Kiểu | Ý nghĩa | Giá trị |
|-----|------|---------|---------|
| **Attrition** | object | Nhân viên đã nghỉ việc hay chưa | `Yes` (nghỉ việc), `No` (ở lại) — tỷ lệ ~16% Yes |

## Thông tin cá nhân

| Cột | Kiểu | Ý nghĩa | Giá trị / Phạm vi |
|-----|------|---------|-------------------|
| EmpID | object | Mã nhân viên (ID duy nhất) | VD: RM297, RM302 |
| Age | int64 | Tuổi nhân viên | 18–60 |
| AgeGroup | object | Nhóm tuổi (phái sinh từ Age) | 18-25, 26-35, 36-45, 46-55, 55+ |
| Gender | object | Giới tính | Male, Female |
| MaritalStatus | object | Tình trạng hôn nhân | Single, Married, Divorced |
| DistanceFromHome | int64 | Khoảng cách từ nhà đến công ty (km) | 1–29 |
| Over18 | object | Trên 18 tuổi (hằng số) | Y (tất cả) |

## Học vấn

| Cột | Kiểu | Ý nghĩa | Giá trị |
|-----|------|---------|---------|
| Education | int64 | Trình độ học vấn | 1=Dưới ĐH, 2=CĐ, 3=Cử nhân, 4=Thạc sĩ, 5=Tiến sĩ |
| EducationField | object | Lĩnh vực học | Life Sciences, Medical, Marketing, Technical Degree, Human Resources, Other |

## Công việc & Tổ chức

| Cột | Kiểu | Ý nghĩa | Giá trị |
|-----|------|---------|---------|
| Department | object | Phòng ban | Research & Development, Sales, Human Resources |
| JobRole | object | Vai trò công việc | 9 vai trò: Laboratory Technician, Sales Representative, Research Scientist, v.v. |
| JobLevel | int64 | Cấp bậc công việc | 1 (thấp nhất) – 5 (cao nhất) |
| BusinessTravel | object | Tần suất đi công tác | Non-Travel, Travel_Rarely, Travel_Frequently |
| OverTime | object | Có làm thêm giờ không | Yes, No |

## Thu nhập & Lương thưởng

| Cột | Kiểu | Ý nghĩa | Giá trị / Phạm vi |
|-----|------|---------|-------------------|
| MonthlyIncome | int64 | Thu nhập hàng tháng (USD) | 1,009 – 19,999 |
| SalarySlab | object | Mức lương (phái sinh) | Upto 5k, 5k-10k, 10k-15k, 15k+ |
| DailyRate | int64 | Lương theo ngày | 102 – 1,499 |
| HourlyRate | int64 | Lương theo giờ | 30 – 100 |
| MonthlyRate | int64 | Tỷ lệ thanh toán hàng tháng | 2,094 – 26,999 |
| PercentSalaryHike | int64 | Phần trăm tăng lương gần nhất | 11 – 25 |
| StockOptionLevel | int64 | Mức cổ phiếu thưởng | 0 – 3 |

## Sự hài lòng & Đánh giá

| Cột | Kiểu | Ý nghĩa | Giá trị |
|-----|------|---------|---------|
| JobSatisfaction | int64 | Mức hài lòng với công việc | 1=Thấp, 2=TB, 3=Cao, 4=Rất cao |
| EnvironmentSatisfaction | int64 | Mức hài lòng với môi trường | 1=Thấp, 2=TB, 3=Cao, 4=Rất cao |
| RelationshipSatisfaction | int64 | Mức hài lòng với quan hệ đồng nghiệp | 1=Thấp, 2=TB, 3=Cao, 4=Rất cao |
| JobInvolvement | int64 | Mức độ gắn bó với công việc | 1=Thấp, 2=TB, 3=Cao, 4=Rất cao |
| WorkLifeBalance | int64 | Cân bằng công việc-cuộc sống | 1=Kém, 2=Tốt, 3=Rất tốt, 4=Xuất sắc |
| PerformanceRating | int64 | Đánh giá hiệu suất | 3=Xuất sắc, 4=Nổi bật (chỉ 2 giá trị) |

## Kinh nghiệm & Thâm niên

| Cột | Kiểu | Ý nghĩa | Giá trị / Phạm vi |
|-----|------|---------|-------------------|
| TotalWorkingYears | int64 | Tổng số năm kinh nghiệm | 0 – 40 |
| YearsAtCompany | int64 | Số năm làm tại công ty hiện tại | 0 – 40 |
| YearsInCurrentRole | int64 | Số năm ở vị trí hiện tại | 0 – 18 |
| YearsSinceLastPromotion | int64 | Số năm kể từ lần thăng chức gần nhất | 0 – 15 |
| YearsWithCurrManager | float64 | Số năm làm việc với quản lý hiện tại | 0 – 17 |
| NumCompaniesWorked | int64 | Số công ty đã từng làm việc | 0 – 9 |
| TrainingTimesLastYear | int64 | Số lần đào tạo trong năm qua | 0 – 6 |

## Cột hệ thống (loại bỏ khi phân tích)

| Cột | Kiểu | Lý do loại bỏ |
|-----|------|---------------|
| EmployeeCount | int64 | Hằng số = 1 (không mang thông tin) |
| EmployeeNumber | int64 | ID nội bộ (trùng với EmpID) |
| StandardHours | int64 | Hằng số = 80 (không mang thông tin) |
| Over18 | object | Hằng số = Y (không mang thông tin) |
| AgeGroup | object | Phái sinh từ Age (dư thừa) |
| SalarySlab | object | Phái sinh từ MonthlyIncome (dư thừa) |

## Ghi chú
- Dataset mất cân bằng: chỉ ~16% nhân viên nghỉ việc (Attrition=Yes)
- `PerformanceRating` chỉ có 2 giá trị (3, 4) → ít khả năng phân biệt
- `YearsWithCurrManager` là cột duy nhất có kiểu float64 (có thể chứa missing values gốc)
- `BusinessTravel` có 4 giá trị do lỗi nhập liệu: "Travel_Rarely" và "TravelRarely" → cần chuẩn hóa khi tiền xử lý
