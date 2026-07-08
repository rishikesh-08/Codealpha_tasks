# Unemployment in India — Analysis

A Python-based analysis of India's estimated unemployment rate across 28 states/UTs, covering Rural and Urban areas from May 2019 to June 2020. The project cleans the raw data, explores trends, and visualizes the impact of the COVID-19 lockdown alongside seasonal patterns.

## Project Structure

```
Task2/
├── Unemployment in India.csv        # raw input dataset
├── unemployment_analysis.py          # main analysis script
├── clean_data.csv                    # generated: cleaned dataset
├── region_period.csv                 # generated: pre/post COVID averages by state
└── charts/                           # generated: output visualizations
    ├── 1_national_trend.png
    ├── 2_rural_vs_urban.png
    ├── 3_state_covid_impact.png
    ├── 4_seasonality.png
    ├── 5_distribution_shift.png
    └── 6_participation_vs_unemployment.png
```

## Requirements

```
pip install pandas matplotlib
```

## How to Run

1. Make sure `Unemployment in India.csv` is in the same folder as the script (or update the `INPUT_CSV` path at the top of `unemployment_analysis.py`).
2. Run:

```
python "unemployment_analysis.py"
```

3. Check the console output for summary stats, and the `charts/` folder for the generated PNG visualizations.

## Data Cleaning Steps

- Stripped leading/trailing whitespace from column names and text fields (a known artifact of this dataset).
- Removed fully-blank rows.
- Parsed `Date` (format `DD-MM-YYYY`) into proper datetime objects.
- Derived `Year`, `Month`, and `Month_Name` fields for seasonal analysis.
- Flagged each row as **Pre-COVID** (May 2019–Mar 2020) or **Post-COVID** (Apr–Jun 2020), split at India's 25 March 2020 nationwide lockdown.

Result: 740 valid monthly observations across 28 states/UTs × 2 area types (Rural, Urban).

## Key Findings

- **National unemployment more than doubled** after the lockdown: 9.6% average pre-COVID → 20.1% post-lockdown, peaking at 24.9% in May 2020.
- **Urban areas were hit harder than rural at the peak** (28.4% vs 21.2% in May 2020), consistent with urban reliance on services/retail/manufacturing versus agriculture's exemption from lockdown restrictions.
- **The shock was highly uneven by state.** Puducherry, Jharkhand, Tamil Nadu, and Bihar saw the steepest increases (+23 to +56 percentage points), while Chandigarh, Jammu & Kashmir, and Himachal Pradesh saw unemployment *fall* over the same window.
- **Labour force participation and unemployment moved together**, not in opposition (correlation ≈ 0.003) — suggesting many people left the workforce entirely during the shock rather than being counted as unemployed, meaning headline unemployment may understate the real disruption.
- A **mild seasonal pattern** (lower mid-year, higher in winter) appears in the pre-COVID data, plausibly tied to agricultural cycles, though one year of data isn't enough to confirm it statistically.
- By **June 2020**, both rural and urban rates showed signs of recovery (converging near 11.8–12.0%) but remained above pre-COVID norms — the dataset ends before full recovery is observed.

## Notes & Limitations

- Data source covers May 2019–June 2020 only; longer-term recovery and any second-wave effects are outside scope.
- Some states have limited/irregular reporting; their averages should be interpreted with caution.
- Seasonal pattern is based on a single pre-COVID year and should be treated as indicative, not confirmed.
