import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")

df = pd.read_csv(r"C:\Users\aryac\OneDrive\Desktop\Task3\car data.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

CURRENT_YEAR = 2026

df["Car_Age"] = CURRENT_YEAR - df["Year"]

df["Brand"] = df["Car_Name"].str.split().str[0].str.lower()

brand_counts = df["Brand"].value_counts()
common_brands = brand_counts[brand_counts >= 3].index
df["Brand"] = df["Brand"].where(df["Brand"].isin(common_brands), "other")

df["Kms_Per_Year"] = df["Driven_kms"] / df["Car_Age"].replace(0, 1)

target = "Selling_Price"
numeric_features = ["Present_Price", "Driven_kms", "Car_Age", "Kms_Per_Year", "Owner"]
categorical_features = ["Brand", "Fuel_Type", "Selling_type", "Transmission"]

X = df[numeric_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
}

results = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "MAE (lakhs)": mae, "RMSE (lakhs)": rmse, "R2 Score": r2})
    fitted_pipelines[name] = pipe

results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_pipeline = fitted_pipelines[best_model_name]
print(f"Best model: {best_model_name}")

rf_pipe = fitted_pipelines["Random Forest"]
feature_names = (
    numeric_features
    + list(rf_pipe.named_steps["preprocess"].named_transformers_["cat"].get_feature_names_out(categorical_features))
)
importances = rf_pipe.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values("Importance", ascending=False).head(12)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=results_df, x="Model", y="R2 Score", hue="Model", legend=False, palette="viridis", ax=ax)
ax.set_title("Model Comparison: R2 Score on Test Set", fontsize=13, fontweight="bold")
ax.set_ylabel("R2 Score")
ax.set_xlabel("")
plt.xticks(rotation=15)
for i, v in enumerate(results_df["R2 Score"]):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

best_preds = best_pipeline.predict(X_test)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, best_preds, alpha=0.6, edgecolor="k", color="#2b6cb0")
lims = [0, max(y_test.max(), best_preds.max()) + 1]
ax.plot(lims, lims, "r--", label="Perfect Prediction")
ax.set_xlabel("Actual Selling Price (lakhs)")
ax.set_ylabel("Predicted Selling Price (lakhs)")
ax.set_title(f"Actual vs Predicted Price - {best_model_name}", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature", hue="Feature", legend=False, palette="mako", ax=ax)
ax.set_title("Feature Importance (Random Forest)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(data=df, x="Car_Age", y="Selling_Price", hue="Fuel_Type", ax=axes[0])
axes[0].set_title("Selling Price vs Car Age")
sns.scatterplot(data=df, x="Present_Price", y="Selling_Price", hue="Transmission", ax=axes[1])
axes[1].set_title("Selling Price vs Present (Showroom) Price")
plt.tight_layout()
plt.savefig("price_relationships.png", dpi=150)
plt.show()

results_df.to_csv("model_comparison_results.csv", index=False)

print("All charts and results saved to current folder")