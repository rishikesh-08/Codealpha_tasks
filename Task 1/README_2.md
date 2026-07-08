# Iris Flower Species Classifier

A machine learning project that classifies iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — using petal and sepal measurements. Three models are trained and compared: Random Forest, Logistic Regression, and SVM.

---

## Project Structure

```
task1/
├── iris_classifier.py   # Main script
├── Iris.csv             # Dataset
└── README.md            # This file
```

---

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn

Install dependencies:

```bash
pip install pandas scikit-learn numpy
```

> If `pip` is not found, try `pip3` or `python -m pip install ...`

---

## Usage

Place `Iris.csv` in the same folder as the script, then run:

```bash
python iris_classifier.py
```

To specify a custom path to the CSV:

```bash
python iris_classifier.py --data path/to/Iris.csv
```

---

## Dataset

The Iris dataset contains **150 samples** across 3 species (50 each), with 4 features per sample:

| Feature | Description |
|---|---|
| SepalLengthCm | Length of the sepal in cm |
| SepalWidthCm | Width of the sepal in cm |
| PetalLengthCm | Length of the petal in cm |
| PetalWidthCm | Width of the petal in cm |

The dataset is split **80/20** into training (120 samples) and test (30 samples) sets.

---

## Models & Results

| Model | Test Accuracy | 5-Fold CV Accuracy |
|---|---|---|
| Random Forest | 90.00% | 94.67% ± 2.49% |
| Logistic Regression | 96.67% | 97.33% ± 2.49% |
| SVM (RBF kernel) | 96.67% | 98.00% ± 1.33% |

**Logistic Regression** and **SVM** both outperform Random Forest on this dataset. For linearly separable data like Iris, simpler models generalise just as well (or better) than complex ensembles.

---

## Feature Importances

Derived from the Random Forest model:

| Feature | Importance |
|---|---|
| PetalWidthCm | 43.7% |
| PetalLengthCm | 43.1% |
| SepalLengthCm | 11.6% |
| SepalWidthCm | 1.5% |

Petal measurements account for ~87% of predictive power. Sepal width contributes almost nothing.

---

## Troubleshooting

**`pip` not found**
```bash
python -m pip install pandas scikit-learn numpy
```

**`UnicodeEncodeError` on Windows**

Run this in your terminal before executing the script:
```bash
set PYTHONIOENCODING=utf-8
```

Or replace the Unicode characters in the script (`─` → `-`, `█` → `#`) to use plain ASCII.

**`python` not recognised**

Python is not installed or not on your PATH. Download it from [python.org](https://www.python.org/downloads/) and check **"Add Python to PATH"** during installation.
