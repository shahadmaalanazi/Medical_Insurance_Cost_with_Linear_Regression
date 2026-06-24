# 🏥 Insurance Charges Prediction

Predicting medical insurance costs using Linear Regression and Random Forest.

---

## 📁 Project Structure

```
├── insurance.csv                   # Raw dataset
├── insurance_analysis_final.py     # Main code
├── eda.jpeg                        # Exploratory analysis plots
├── models.jpeg                      # Model comparison plots
└── README.md
```

---

## 📊 Dataset

| Column | Type | Description |
|---|---|---|
| `age` | int | Age of the beneficiary |
| `sex` | str | Gender (male / female) |
| `bmi` | float | Body Mass Index |
| `children` | int | Number of children |
| `smoker` | str | Smoking status (yes / no) |
| `region` | str | Geographic region |
| `charges` | float | Insurance cost ⬅️ Target |

- **Size:** 1,338 rows × 7 columns
- **Duplicates:** 1 row removed
- **Missing values:** None

---

## ⚙️ Pipeline

**1. Data Cleaning**
- Remove duplicate rows
- Check for missing values

**2. Exploratory Data Analysis (EDA)**
- Charges distribution
- Impact of smoking on charges
- BMI vs charges relationship
- Average charges by region and age
- Correlation heatmap

**3. Preprocessing**
- Encode categorical variables with `get_dummies`
- Normalize numerical features with `StandardScaler`
- Train/Test split: 80% / 20%

**4. Models**
- Linear Regression
- Random Forest (100 estimators)

**5. Evaluation**
- 5-Fold Cross-Validation

---

## 📈 Evaluation Metrics

> Proper regression metrics are used — Precision & Recall are classification metrics only.

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — same unit as target |
| **RMSE** | Root Mean Squared Error — penalizes large errors more |
| **R²** | Proportion of variance explained — higher is better (max = 1) |
| **MAPE** | Mean Absolute Percentage Error |

---

## 🏆 Results

| Model | R² (Test) | RMSE (Test) | MAE (Test) |
|---|---|---|---|
| Linear Regression | 0.807 | 5,956 | 4,177 |
| **Random Forest** | **0.877** | **4,764** | **2,713** |

**Random Forest outperforms Linear Regression across all metrics.**  
> Note: Random Forest R² train = 0.975 vs test = 0.877 → slight overfitting; can be improved by tuning hyperparameters.

---

## 🚀 How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python insurance_analysis_final.py
```

---

## 🛠️ Libraries

- `pandas` — data manipulation
- `numpy` — numerical operations
- `matplotlib` / `seaborn` — visualizations
- `scikit-learn` — models and evaluation
