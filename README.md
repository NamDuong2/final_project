# Traffic Safety and Air Quality Across NYC Boroughs

CSE 163 Final Project — Hoai Nam Duong

## Necessary Downloads and Installations

This project was written in **Python 3.10**. You will need the following libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

If you have `pip` set up, you can install everything at once by running:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

No virtual environment file (`environment.yml` / `requirements.txt`) is included in this
repository, the packages above are the only dependencies beyond the Python standard library.

### Data downloads

This project uses three datasets and one of them is collisions data that are **not included in this repository** because of
its huge size. Download the that one and place it directly in the same folder as
`main.py`, `ml_pipeline.py`, and `test.py` (do **not** put them in a subfolder because the code
loads them by filename only, e.g. `"collision_crashes.csv"`):

1. **NYC Motor Vehicle Collisions – Crashes**
   Source: [https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data)
   Download as CSV and rename the file to `collision_crashes.csv`.

2. **NYC Air Quality Surveillance Data**
   Source: [https://data.cityofnewyork.us/Environment/Air-Quality/c3uy-2p5r](https://data.cityofnewyork.us/Environment/Air-Quality-and-Health-Impacts/c3uy-2p5r/about_data)
   Already included and named as `air_quality.csv`.

3. **NYC Neighborhood Poverty**
   Source: [https://a816-dohbesp.nyc.gov/IndicatorPublic/data-explorer/economic-conditions/?id=103#display=summary](https://a816-dohbesp.nyc.gov/IndicatorPublic/data-explorer/economic-conditions/?id=103#display=summary)
   Already included and named as `neighborhood_poverty.csv`.

After this step, your project folder should contain:

```
main.py
ml_pipeline.py
test.py
collision_crashes.csv
air_quality.csv
neighborhood_poverty.csv
```

## File Descriptions

- **`main.py`:**  The main entry point for the project. Running this file executes the full
  pipeline in order:
  1. Loads and cleans the collisions dataset (drops rows missing critical columns, adds
     `CRASH HOUR` and `SEVERE_CRASH` columns).
  2. Loads and filters the air quality dataset down to Borough-level summer-mean ozone
     readings.
  3. Loads the poverty dataset and adds a `Poverty_group` (Higher-Poverty / Lower-Poverty)
     column based on a median split.
  4. Joins collisions with air quality on `(BOROUGH_ID, YEAR)` to build the RQ3 feature
     table, then calls into `ml_pipeline.py` to train and evaluate the RQ3 classifiers.
  5. Expands the poverty dataset's multi-year time periods into individual years and joins
     it with the RQ3 table to build the RQ4 table.
  6. Calls `test.py` to run all sanity-check assertions on the cleaned/joined data.

  All of the `print(...)` statements produce the missing-data summaries, 7-number summaries,
  and value counts referenced in the report. Plotting is **OFF by default**, see more
  "Reproducing the figures" below.

- **`ml_pipeline.py`:** Contains `run_rq3_machine_learning()`, which is called from
  `main.py`. This function engineers the RQ3 features (day of week, crash hour, borough,
  vehicle type, contributing factor, ozone level), splits the data into train/test sets,
  trains three classifiers (Logistic Regression, Random Forest, Gradient Boosting) inside a
  `scikit-learn` `Pipeline` with one-hot encoding/scaling, prints a precision/recall/F1/ROC-AUC
  comparison table, and prints the top 10 most important features from the Gradient Boosting
  model.

- **`test.py`:** Contains `run_sanity_checks()`, called at the end of `main.py`. Runs `assert`
  statements that verify: `Unknown` boroughs were filtered out, critical columns have no
  missing values, high-missingness columns were dropped, latitude/longitude were imputed,
  `CRASH HOUR` values fall between 0–23, ozone values are non-negative and within a valid year
  range, poverty percentages fall between 0–100, and that the RQ3/RQ4 joins did not create
  duplicate rows (Cartesian explosion).

## Step-by-Step Instructions to Run the Project

1. Make sure the three `.py` files and the three renamed CSV files are all in the same
   folder (see "Data downloads" above).
2. Open a terminal and navigate to that folder.
3. Run:
   ```
   python main.py
   ```
4. The script will print, in order: the RQ1/RQ2 collision cleaning summary (currently
   commented out by default — see note below), the air quality and poverty missing-data
   checks (also commented out by default), the RQ3 joined-table summary, the RQ3 machine
   learning performance table and top-10 feature importances, the RQ4 joined-table summary,
   and finally `"ALL DEEP-DIVE DATASET TESTS PASSED!"` if every sanity check succeeds.

   Expect the machine learning step to take a couple of minutes, since it trains three
   models (including a Gradient Boosting classifier) on a 100,000-row sample of the joined
   dataset.

### Reproducing the figures

Figures 1–8 from the report are **not generated by default**, setting via `RUN_PLOTS` at the top of `main.py`:

```python
RUN_PLOTS = False
```

To regenerate all figures, open `main.py`, change this line to `RUN_PLOTS = True`, and re-run
`python main.py`. Each figure will be saved as a `.png` file (e.g. `figure_1.png` through
`figure_8.png`) in the same folder, except for the RQ3 boxplot and the RQ4 `lmplot`, which
will also pop up in a separate window via `plt.show()` in addition to being saved.


## Notes

- All file paths in the code are relative filenames with no folder prefix, so the script
  must be run from inside the project folder as described above. 
- The `ml_pipeline.py` step automatically samples the joined RQ3 dataset to 100,000 rows
  before splitting into train/test sets, to keep training time reasonable, this is
  expected behavior, not a bug.
- If any `assert` in `test.py` fails, the script will stop and print which specific check
  failed.
- When crash occur, this most likely means one of the source CSVs has a different schema/column
  naming than expected (for example, if the City of New York changes column names in a future
  export).
