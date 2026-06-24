# ============================================================
#  Insurance Charges Prediction & Analysis
#  Dataset: insurance.csv  (1338 rows × 7 cols)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


# ============================================================
# 1. LOAD & CLEAN DATA
# ============================================================

data = pd.read_csv('insurance.csv')   # عدّل المسار حسب موقع الملف

print("=" * 55)
print("Dataset Info")
print("=" * 55)
print(f"Shape       : {data.shape}")
print(f"Duplicates  : {data.duplicated().sum()}")
print(f"Missing     : {data.isnull().sum().sum()}")

data = data.drop_duplicates()
print(f"Shape after cleaning: {data.shape}")
print()
print(data.describe().round(2))


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================

sns.set(style='whitegrid')

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Insurance Data — Exploratory Analysis', fontsize=16, fontweight='bold')

# توزيع التكاليف
sns.histplot(data['charges'], bins=30, kde=True, color='steelblue', ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Charges')
axes[0, 0].set_xlabel('Charges')
axes[0, 0].set_ylabel('Frequency')

# التكاليف حسب التدخين
sns.boxplot(data=data, x='smoker', y='charges', hue='smoker',
            palette='coolwarm', legend=False, ax=axes[0, 1])
axes[0, 1].set_title('Charges by Smoking Status')

# التكاليف مقابل BMI مع التدخين
sns.scatterplot(data=data, x='bmi', y='charges',
                hue='smoker', palette='coolwarm', alpha=0.7, ax=axes[0, 2])
axes[0, 2].set_title('Charges vs BMI (by Smoker)')

# متوسط التكاليف حسب المنطقة والتدخين
sns.barplot(data=data, x='region', y='charges',
            hue='smoker', palette='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('Avg Charges by Region & Smoker')
axes[1, 0].tick_params(axis='x', rotation=15)

# التكاليف مقابل العمر
sns.regplot(data=data, x='age', y='charges',
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[1, 1])
axes[1, 1].set_title('Charges vs Age')

# خريطة الارتباط
corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1, 2])
axes[1, 2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.show()


# ============================================================
# 3. PREPROCESSING
# ============================================================

X = data.drop('charges', axis=1)
y = data['charges']

# تحويل المتغيرات النصية إلى أرقام
X_enc = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)

# تطبيع المتغيرات الرقمية
scaler = StandardScaler()
X_enc[['age', 'bmi', 'children']] = scaler.fit_transform(X_enc[['age', 'bmi', 'children']])

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, random_state=42
)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")


# ============================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    """تدريب النموذج وطباعة مقاييس الأداء الصحيحة للريقريشن."""
    model.fit(X_train, y_train)
    y_tr = model.predict(X_train)
    y_te = model.predict(X_test)

    results = {
        'MAE_train':  metrics.mean_absolute_error(y_train, y_tr),
        'MAE_test':   metrics.mean_absolute_error(y_test,  y_te),
        'RMSE_train': np.sqrt(metrics.mean_squared_error(y_train, y_tr)),
        'RMSE_test':  np.sqrt(metrics.mean_squared_error(y_test,  y_te)),
        'R2_train':   metrics.r2_score(y_train, y_tr),
        'R2_test':    metrics.r2_score(y_test,  y_te),
        'MAPE_test':  np.mean(np.abs((y_test - y_te) / y_test)) * 100,
    }

    print(f"\n{name}")
    print(f"  MAE   → Train: {results['MAE_train']:>9,.2f}  | Test: {results['MAE_test']:>9,.2f}")
    print(f"  RMSE  → Train: {results['RMSE_train']:>9,.2f}  | Test: {results['RMSE_test']:>9,.2f}")
    print(f"  R²    → Train: {results['R2_train']:>9.3f}  | Test: {results['R2_test']:>9.3f}")
    print(f"  MAPE  → Test : {results['MAPE_test']:>9.2f}%")

    return model, y_tr, y_te, results


print("\n" + "=" * 55)
print("Model Results")
print("=" * 55)

lin_reg, y_tr_lr, y_te_lr, res_lr = evaluate_model(
    LinearRegression(), X_train, X_test, y_train, y_test, "Linear Regression"
)

rfr, y_tr_rf, y_te_rf, res_rf = evaluate_model(
    RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1),
    X_train, X_test, y_train, y_test, "Random Forest"
)


# ============================================================
# 5. CROSS-VALIDATION
# ============================================================

cv_rmse = -cross_val_score(
    rfr, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error'
)
print(f"\nRandom Forest — CV RMSE (5-fold): {cv_rmse.round(0)}")
print(f"Mean: {cv_rmse.mean():,.2f}  |  Std: {cv_rmse.std():,.2f}")


# ============================================================
# 6. MODEL COMPARISON PLOT
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Actual vs Predicted Charges', fontsize=14, fontweight='bold')

for ax, name, y_tr, y_te, res in [
    (axes[0], "Linear Regression", y_tr_lr, y_te_lr, res_lr),
    (axes[1], "Random Forest",     y_tr_rf, y_te_rf, res_rf),
]:
    ax.scatter(y_train, y_tr, c='gray',      s=20, alpha=0.4, label='Train')
    ax.scatter(y_test,  y_te, c='royalblue', s=20, alpha=0.7, label='Test')
    lim = [0, max(y_test.max(), y_te.max())]
    ax.plot(lim, lim, 'r--', lw=1.5, label='Perfect Fit')
    ax.set_title(f"{name}\nR²={res['R2_test']:.3f}  |  RMSE={res['RMSE_test']:,.0f}")
    ax.set_xlabel('Actual Charges')
    ax.set_ylabel('Predicted Charges')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
