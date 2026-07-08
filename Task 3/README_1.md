# Car Price Prediction with Machine Learning

Predicts used-car selling prices from features like brand, age, mileage, fuel type, and transmission using regression models.

## Overview

This project trains and compares several regression models to predict the selling price of used cars based on historical listing data. It covers the full ML workflow: data preprocessing, feature engineering, model training, evaluation, and visualization.

## Dataset

`car data.csv` — 301 used car listings with the following columns:

| Column | Description |
|---|---|
| `Car_Name` | Model name (used to derive `Brand`) |
| `Year` | Manufacturing year |
| `Selling_Price` | Target variable — price the car was sold for (in lakhs) |
| `Present_Price` | Current showroom price of a new model (in lakhs) |
| `Driven_kms` | Total kilometers driven |
| `Fuel_Type` | Petrol / Diesel / CNG |
| `Selling_type` | Dealer / Individual |
| `Transmission` | Manual / Automatic |
| `Owner` | Number of previous owners |

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run

1. Make sure `car data.csv` and `cardata.py` are in the same folder.
2. Run:
   ```bash
   python cardata.py
   ```
3. Charts and results will be saved to the same folder:
   - `model_comparison.png`
   - `actual_vs_predicted.png`
   - `feature_importance.png`
   - `price_relationships.png`
   - `model_comparison_results.csv`

## Feature Engineering

Since the raw data doesn't include direct measures like horsepower, the following engineered features act as proxies:

- **`Car_Age`** — computed from `Year`; a stand-in for depreciation
- **`Brand`** — extracted from the first word of `Car_Name`; rare brands (fewer than 3 listings) are grouped into `other` to avoid overfitting
- **`Kms_Per_Year`** — usage intensity (`Driven_kms` / `Car_Age`), often more informative than raw mileage
- **`Present_Price`** — used as-is; the strongest available proxy for brand goodwill and market positioning

## Models Trained

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Each model is trained inside a `scikit-learn` `Pipeline` with a shared preprocessing step (standard scaling for numeric features, one-hot encoding for categorical features), then evaluated on a held-out 20% test split.

## Results

| Model | MAE (lakhs) | RMSE (lakhs) | R² Score |
|---|---|---|---|
| **Gradient Boosting** | 0.55 | 0.91 | **0.964** |
| Random Forest | 0.63 | 0.96 | 0.960 |
| Lasso Regression | 1.06 | 1.64 | 0.884 |
| Linear Regression | 1.09 | 1.65 | 0.881 |
| Ridge Regression | 1.09 | 1.66 | 0.881 |

**Best model:** Gradient Boosting Regressor, explaining ~96% of the variance in selling price with an average error of ~0.55 lakhs.

**Key insight:** `Present_Price` (showroom price of the current equivalent model) is by far the strongest predictor — resale value is heavily anchored to what a brand-new version of the car costs today. `Car_Age` and `Kms_Per_Year` are the next most important signals.

## Files

- `cardata.py` — main script: loads data, engineers features, trains/compares models, generates charts
- `car data.csv` — source dataset
- `model_comparison_results.csv` — output: metrics table for all trained models
- `model_comparison.png` — bar chart comparing R² scores across models
- `actual_vs_predicted.png` — scatter plot of actual vs. predicted prices for the best model
- `feature_importance.png` — feature importance ranking from the Random Forest model
- `price_relationships.png` — exploratory scatter plots (price vs. age, price vs. showroom price)

## Possible Extensions

- Add real horsepower/engine specs if a richer dataset becomes available
- Hyperparameter tuning (e.g. `GridSearchCV`) for the Gradient Boosting model
- Try log-transforming `Selling_Price` to handle the right-skewed distribution of high-end cars
