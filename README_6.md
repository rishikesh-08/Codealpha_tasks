# CodeAlpha Data Science Internship — Projects

A collection of end-to-end machine learning projects completed as part of the CodeAlpha Data Science Internship. Each task covers the full ML workflow: data cleaning, feature engineering, model training and comparison, evaluation, and visualization.

## Projects

| Task | Project | Type | Best Model / Method | Score |
|---|---|---|---|---|
| Task1 | [Iris Flower Classification](#task1--iris-flower-classification) | Classification | SVM / Logistic Regression | 96.67% accuracy |
| Task2 | [Unemployment in India Analysis](#task2--unemployment-in-india-analysis) | Exploratory Data Analysis | — | — |
| Task3 | [Car Price Prediction](#task3--car-price-prediction) | Regression | Gradient Boosting | R² = 0.964 |
| Task4 | [Sales Prediction (Advertising)](#task4--sales-prediction-using-advertising-data) | Regression | Polynomial Regression (deg 2) | R² = 0.987 |

---

## Task1 — Iris Flower Classification

Classifies iris flowers into Setosa, Versicolor, or Virginica using petal and sepal measurements.

**Dataset:** 150 samples, 4 features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`).

**Models compared:** Random Forest, Logistic Regression, SVM (RBF kernel).

**Result:** Logistic Regression and SVM both reached 96.67% test accuracy (SVM: 98.00% ± 1.33% on 5-fold CV), outperforming Random Forest (90.00%). Petal measurements accounted for ~87% of predictive power.

**Folder contents:**
```
Task1/
├── Iris.csv              # dataset
├── iris_classifier.py    # main script
├── output.txt            # console output log
└── README_2.md           # project documentation
```

```bash
cd Task1
pip install pandas scikit-learn numpy
python iris_classifier.py
```

---

## Task2 — Unemployment in India Analysis

A Python-based analysis of India's estimated unemployment rate across 28 states/UTs, covering Rural and Urban areas from May 2019 to June 2020. Cleans the raw data, explores trends, and visualizes the impact of the COVID-19 lockdown alongside seasonal patterns.

**Dataset:** 740 valid monthly observations across 28 states/UTs × 2 area types (Rural, Urban).

**Data cleaning:** Whitespace stripped from columns/text, blank rows removed, dates parsed, Year/Month fields derived, rows flagged as Pre-COVID (May 2019–Mar 2020) or Post-COVID (Apr–Jun 2020).

**Key findings:**
- National unemployment more than doubled after lockdown: 9.6% average pre-COVID → 20.1% post-lockdown, peaking at 24.9% in May 2020.
- Urban areas were hit harder than rural at the peak (28.4% vs 21.2% in May 2020).
- Impact was highly uneven by state — Puducherry, Jharkhand, Tamil Nadu, and Bihar saw the steepest increases (+23 to +56 pp), while Chandigarh, Jammu & Kashmir, and Himachal Pradesh saw unemployment *fall*.
- Labour force participation and unemployment moved together (correlation ≈ 0.003), not in opposition — suggesting many people left the workforce entirely rather than being counted as unemployed.
- A mild seasonal pattern (lower mid-year, higher in winter) appears in the pre-COVID data, though one year isn't enough to confirm it statistically.
- By June 2020, both rural and urban rates showed signs of recovery (converging near 11.8–12.0%) but remained above pre-COVID norms.

**Folder contents:**
```
Task2/
├── Unemployment in India.csv    # raw dataset
├── unemployment_analysis.py     # main analysis script
├── clean_data.csv               # generated: cleaned dataset
├── region_period.csv            # generated: pre/post COVID averages by state
├── outout                       # console output log
├── charts/                      # generated visualizations
│   ├── 1_national_trend.png
│   ├── 2_rural_vs_urban.png
│   ├── 3_state_covid_impact.png
│   ├── 4_seasonality.png
│   ├── 5_distribution_shift.png
│   └── 6_participation_vs_unemployment.png
└── README_5.md                  # project documentation
```

```bash
cd Task2
pip install pandas matplotlib
python unemployment_analysis.py
```

---

## Task3 — Car Price Prediction

Predicts the resale (selling) price of used cars from features like showroom price, age, mileage, fuel type, and transmission.

**Dataset:** 301 used car listings (`car data.csv`) — `Car_Name`, `Year`, `Present_Price`, `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission`, `Owner`.

**Feature engineering:** Brand extracted from car name, car age, kilometers driven per year.

**Models compared:** Linear, Ridge, Lasso, Random Forest, Gradient Boosting.

**Result:** Gradient Boosting performed best — R² of 0.964, MAE of 0.55 lakhs. `Present_Price` was the strongest predictor, followed by car age and usage intensity.

**Folder contents:**
```
Task3/
├── car data.csv                  # dataset
├── cardata.py                    # main script
├── model_comparison_results.csv  # metrics table output
├── charts/                       # regression graphs (model comparison, actual vs predicted, feature importance, etc.)
└── README_1.md                   # project documentation
```

```bash
cd Task3
pip install pandas numpy matplotlib seaborn scikit-learn
python cardata.py
```

---

## Task4 — Sales Prediction Using Advertising Data

Predicts sales from TV, Radio, and Newspaper advertising spend, and quantifies how shifting budget between channels affects sales.

**Dataset:** 200 rows, no missing values, no duplicates (`Advertising.csv`) — `TV`, `Radio`, `Newspaper`, `Sales`. 2 outliers flagged in `Newspaper` (not auto-removed).

**Feature engineering:** Total spend, per-channel spend share, TV×Radio interaction term.

**Correlation with Sales:** TV 0.78, Radio 0.58, Newspaper 0.23, Total Spend 0.87 — advertising spend as a whole correlates more strongly with sales than any single channel.

**Models compared** (80/20 split, 5-fold cross-validation):

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.899 | 1.46 | 1.78 |
| Random Forest | 0.977 | 0.68 | 0.86 |
| **Polynomial Regression (deg 2)** | **0.987** | **0.53** | **0.64** |

Cross-validated R²: Linear 0.887 ± 0.040, Random Forest 0.970 ± 0.009.

**Result:** Polynomial Regression (degree 2) performed best, capturing the TV×Radio synergy effect. Random Forest feature importance: TV 62.9%, Radio 36.1%, Newspaper 1.0% — Newspaper spend has almost no impact on sales.

**Budget-shift simulation:** A +10% increase in spend, applied independently, lifts average predicted sales (baseline ≈ 14.02) by:
- TV: +0.45
- Radio: +0.46
- Newspaper: +0.02

TV and Radio give comparable returns per 10% budget increase; Newspaper spend is essentially not worth increasing.

**Folder contents:**
```
Task4/
├── Advertising.csv    # dataset
├── advert.py          # main script
├── Output.csv         # console output log
├── report_data.json   # all computed metrics (correlations, model scores, coefficients, budget-shift scenarios)
├── charts/            # regression graphs (correlation heatmap, actual vs predicted, model comparison, feature importance, etc.)
└── README.md          # project documentation
```

```bash
cd Task4
pip install pandas numpy matplotlib seaborn scikit-learn
python advert.py
```

---

## Common Workflow

Each project follows the same structure:

1. **Data cleaning** — handle nulls, duplicates, outliers
2. **Feature engineering** — derive proxies for signals not directly in the raw data
3. **Model training** — compare multiple algorithms under a shared preprocessing pipeline
4. **Evaluation** — held-out test set plus cross-validation where applicable
5. **Visualization** — charts saved to each project's `charts/` folder (comparisons, actual-vs-predicted, feature importance, distributions)

## Tech Stack

- Python
- Pandas, NumPy — data handling and preprocessing
- Scikit-learn — models, pipelines, evaluation
- Matplotlib, Seaborn — visualization

## Repository Structure

```
Codealpha_tasks/
├── Task1/    (Iris Flower Classification)
├── Task2/    (Unemployment in India Analysis)
├── Task3/    (Car Price Prediction)
├── Task4/    (Sales Prediction — Advertising)
└── README.md   (this file)
```

## About

These projects were completed as part of the **CodeAlpha Data Science Internship**.
