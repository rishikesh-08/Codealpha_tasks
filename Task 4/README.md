# Sales Prediction Using Advertising Data

Predicts sales from TV, Radio, and Newspaper advertising spend using regression models, and quantifies how shifting budget between channels affects sales.

## Requirements

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Setup

1. Place `analysis.py` in a folder along with your advertising CSV file.
2. Open `analysis.py` and set `FILE_PATH` (near the top) to your CSV's filename or full path.
3. Your CSV must have these columns: `TV`, `Radio`, `Newspaper`, `Sales` (an unnamed index column, as in the original dataset, is fine).

## Run

```
python analysis.py
```

## Output

- `charts/` — 6 PNG charts: correlation heatmap, spend-vs-sales scatterplots, distributions, actual-vs-predicted, model comparison, feature importance
- `report_data.json` — all computed metrics (correlations, model scores, coefficients, budget-shift scenarios) in one file
- Console output — same JSON, printed for a quick look without opening the file

## What it does

1. **Cleans** the data — drops duplicates/nulls, flags outliers via IQR (doesn't auto-remove them)
2. **Engineers features** — total spend, each channel's share of spend, TV×Radio interaction term
3. **Selects features** — correlation of each channel with Sales
4. **Trains 3 models** on an 80/20 split, validated with 5-fold cross-validation:
   - Linear Regression
   - Random Forest
   - Polynomial Regression (degree 2, captures TV×Radio synergy)
5. **Simulates budget shifts** — predicts the sales impact of a +10% spend increase in each channel independently

## Notes

- `FILE_PATH` and `OUTPUT_DIR` are the only settings you should need to change.
- Random seeds are fixed (`random_state=42`) so results are reproducible run to run.
