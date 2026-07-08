import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import os

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120

FILE_PATH = "Advertising.csv"
OUTPUT_DIR = "charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(FILE_PATH, index_col=0)
df.columns = [c.strip() for c in df.columns]

report = {}
report['n_rows'] = len(df)
report['n_missing'] = int(df.isnull().sum().sum())
report['n_duplicates'] = int(df.duplicated().sum())

df = df.drop_duplicates()
df = df.dropna()

outlier_counts = {}
for col in ['TV','Radio','Newspaper','Sales']:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outlier_counts[col] = int(((df[col] < lo) | (df[col] > hi)).sum())
report['outlier_counts'] = outlier_counts

df['Total_Spend'] = df['TV'] + df['Radio'] + df['Newspaper']
df['TV_share'] = df['TV'] / df['Total_Spend']
df['Radio_share'] = df['Radio'] / df['Total_Spend']
df['Newspaper_share'] = df['Newspaper'] / df['Total_Spend']
df['TV_Radio_interaction'] = df['TV'] * df['Radio']

corr = df[['TV','Radio','Newspaper','Total_Spend','Sales']].corr()
report['correlation_with_sales'] = corr['Sales'].drop('Sales').to_dict()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, cbar_kws={'shrink':0.8})
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_correlation_heatmap.png')
plt.close()

fig, axes = plt.subplots(1,3, figsize=(15,4.5))
for ax, col, color in zip(axes, ['TV','Radio','Newspaper'], ['#2563eb','#16a34a','#dc2626']):
    sns.regplot(x=col, y='Sales', data=df, ax=ax, color=color, scatter_kws={'alpha':0.5,'s':25}, line_kws={'color':'black'})
    ax.set_title(f'{col} Spend vs Sales')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_spend_vs_sales.png')
plt.close()

fig, axes = plt.subplots(1,4, figsize=(16,3.5))
for ax, col in zip(axes, ['TV','Radio','Newspaper','Sales']):
    sns.histplot(df[col], kde=True, ax=ax, color='#4f46e5')
    ax.set_title(f'{col} Distribution')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_distributions.png')
plt.close()

features = ['TV','Radio','Newspaper']
X = df[features]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {}

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
models['Linear Regression'] = {
    'R2': r2_score(y_test, pred_lr),
    'MAE': mean_absolute_error(y_test, pred_lr),
    'RMSE': np.sqrt(mean_squared_error(y_test, pred_lr)),
    'coefficients': dict(zip(features, lr.coef_)),
    'intercept': lr.intercept_
}

rf = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=5)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
models['Random Forest'] = {
    'R2': r2_score(y_test, pred_rf),
    'MAE': mean_absolute_error(y_test, pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, pred_rf)),
    'feature_importance': dict(zip(features, rf.feature_importances_))
}

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
lr_poly = LinearRegression()
lr_poly.fit(X_poly_train, y_train)
pred_poly = lr_poly.predict(X_poly_test)
models['Polynomial Regression (deg 2)'] = {
    'R2': r2_score(y_test, pred_poly),
    'MAE': mean_absolute_error(y_test, pred_poly),
    'RMSE': np.sqrt(mean_squared_error(y_test, pred_poly)),
}

cv_scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2')
report['cv_r2_linear'] = {'mean': cv_scores.mean(), 'std': cv_scores.std(), 'scores': cv_scores.tolist()}

cv_scores_rf = cross_val_score(RandomForestRegressor(n_estimators=300, random_state=42, max_depth=5), X, y, cv=5, scoring='r2')
report['cv_r2_rf'] = {'mean': cv_scores_rf.mean(), 'std': cv_scores_rf.std()}

report['models'] = models

plt.figure(figsize=(6,5.5))
plt.scatter(y_test, pred_poly, alpha=0.7, color='#4f46e5', edgecolor='white', s=60)
lims = [min(y_test.min(), pred_poly.min())-1, max(y_test.max(), pred_poly.max())+1]
plt.plot(lims, lims, 'k--', linewidth=1)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title(f'Actual vs Predicted Sales (Polynomial Model, R²={models["Polynomial Regression (deg 2)"]["R2"]:.3f})')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_actual_vs_predicted.png')
plt.close()

plt.figure(figsize=(7,4.5))
names = list(models.keys())
r2s = [models[n]['R2'] for n in names]
bars = plt.bar(names, r2s, color=['#2563eb','#16a34a','#f59e0b'])
plt.ylabel('R² Score (Test Set)')
plt.title('Model Comparison')
plt.ylim(0,1.05)
for b, v in zip(bars, r2s):
    plt.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.3f}', ha='center', fontweight='bold')
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_model_comparison.png')
plt.close()

fig, axes = plt.subplots(1,2, figsize=(11,4.5))
coefs = models['Linear Regression']['coefficients']
axes[0].barh(list(coefs.keys()), list(coefs.values()), color='#2563eb')
axes[0].set_title('Linear Regression Coefficients\n(Sales lift per $1000 extra spend)')
imp = models['Random Forest']['feature_importance']
axes[1].barh(list(imp.keys()), list(imp.values()), color='#16a34a')
axes[1].set_title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_feature_importance.png')
plt.close()

scenarios = {}
base_pred = lr_poly.predict(poly.transform(X))
scenarios['baseline_avg_predicted_sales'] = float(base_pred.mean())

for channel in features:
    X_scenario = X.copy()
    X_scenario[channel] = X_scenario[channel] * 1.10
    pred_scenario = lr_poly.predict(poly.transform(X_scenario))
    lift = pred_scenario.mean() - base_pred.mean()
    scenarios[f'{channel}_+10pct_avg_sales_lift'] = float(lift)

report['scenarios'] = scenarios

with open('report_data.json','w') as f:
    json.dump(report, f, indent=2, default=str)

print(json.dumps(report, indent=2, default=str))