"""
Author: Nam Duong
CSE 163 AA
NYC traffic collision EDA

The script has 4 main stages:
    1. Load & clean the collisions dataset.
    2. Load & filter the air quality dataset.
    3. Load & process the neighborhood poverty dataset.
    4. Join the tables together to answer RQ3 and RQ4.

The plotting sections (matplotlib/seaborn) are split into their own
functions and are NOT run by default (only when RUN_PLOTS = True), to
keep the main pipeline fast.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import test
from ml_pipeline import run_rq3_machine_learning


# Set to True to actually render the figures.
RUN_PLOTS = False

BOROUGH_MAPPING = {
    "BRONX": 1,
    "BROOKLYN": 2,
    "MANHATTAN": 3,
    "QUEENS": 4,
    "STATEN ISLAND": 5,
}


# ---------------------------------------------------------------------------
# 1. COLLISIONS
# ---------------------------------------------------------------------------
def load_collisions(path="collision_crashes.csv"):
    """Load the raw collisions CSV."""
    return pd.read_csv(path, dtype={"ZIP CODE": str})


def clean_collisions(collisions):
    """Clean the collisions data: drop rows missing critical columns,
    and add CRASH HOUR (hour of day) and SEVERE_CRASH (injury/fatality
    flag) columns.
    """
    critical_cols = [
        "VEHICLE TYPE CODE 1",
        "CONTRIBUTING FACTOR VEHICLE 1",
        "BOROUGH",
        "CRASH DATE",
    ]
    collisions_clean = collisions.dropna(subset=critical_cols).copy()

    collisions_clean["CRASH HOUR"] = pd.to_datetime(
        collisions_clean["CRASH TIME"], format="%H:%M", errors="coerce"
    ).dt.hour

    collisions_clean["SEVERE_CRASH"] = (
        collisions_clean["NUMBER OF PERSONS INJURED"]
        + collisions_clean["NUMBER OF PERSONS KILLED"]
    ) > 0

    return collisions_clean


def summarize_collisions(collisions, collisions_clean):
    """Print descriptive stats (missing data, 7-number summary, etc.)."""
    print("=== MISSING DATA CHECK: collisions ===")
    total_rows, total_cols = collisions.shape
    print(f"For collision data -> total rows: {total_rows}, "
          f"total columns: {total_cols}")

    missing_count = collisions.isnull().sum()
    missing_percent = (collisions.isnull().sum() / total_rows) * 100
    missing_summary = pd.DataFrame(
        {
            "Missing Rows": missing_count,
            "Missing Percentage (%)": missing_percent,
        }
    )

    print("=== SUMMARY OF VARIABLES OF INTEREST (collisions) ===")
    quant_cols = ["NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED"]
    seven_num_summary = collisions_clean[quant_cols].describe(
        percentiles=[0.25, 0.50, 0.75]
    )
    print("--- QUANTITATIVE VARIABLES (7-NUMBER SUMMARY) ---")
    print(seven_num_summary)

    for col in ["BOROUGH"]:
        print(f"\n--- CATEGORICAL SUMMARY: {col} ---")
        print(collisions_clean[col].value_counts(dropna=False))

    missing_summary = missing_summary[
        missing_summary["Missing Rows"] > 0
    ].sort_values(by="Missing Percentage (%)", ascending=False)
    print(missing_summary)


def plot_collisions(collisions_clean):
    """Descriptive plots for the collisions analysis (only run when
    RUN_PLOTS is True).
    """
    print("=== VISUALIZATION ===")

    collisions_clean["YEAR"] = pd.to_datetime(
        collisions_clean["CRASH DATE"]
    ).dt.year
    collisions_clean["IS_FATAL"] = (
        collisions_clean["NUMBER OF PERSONS KILLED"] > 0
    )

    # Figure 1: annual trend of fatal crashes
    yearly_fatalities = (
        collisions_clean.groupby("YEAR")["IS_FATAL"].sum().reset_index()
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        yearly_fatalities["YEAR"],
        yearly_fatalities["IS_FATAL"],
        marker="o",
        color="crimson",
        linewidth=2,
        label="Fatal Crashes",
    )
    plt.axvline(
        x=2014, color="black", linestyle="--",
        label="Vision Zero Launch (2014)",
    )
    plt.title("Annual Trend of Fatal Motor Vehicle Crashes in NYC")
    plt.xlabel("Year")
    plt.ylabel("Number of Fatal Crashes")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("figure_1.png", dpi=300, bbox_inches="tight")
    print("Saved plot successfully to figure_1.png!")

    # Figure 2: fatal crashes per year, broken down by borough
    borough_yearly = (
        collisions_clean[collisions_clean["BOROUGH"] != "Unknown"]
        .groupby(["YEAR", "BOROUGH"])["IS_FATAL"]
        .sum()
        .unstack()
    )

    plt.figure(figsize=(11, 6))
    for borough in borough_yearly.columns:
        plt.plot(
            borough_yearly.index,
            borough_yearly[borough],
            marker="o",
            linewidth=2,
            label=borough,
        )
    plt.axvline(
        x=2014, color="black", linestyle="--", alpha=0.7,
        label="Vision Zero (2014)",
    )
    plt.title("Annual Fatal Crashes by Borough Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Fatal Crashes")
    plt.legend(title="Borough")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("figure_2.png", dpi=300, bbox_inches="tight")
    print("Saved plot successfully to figure_2.png!")

    # Figure 3: frequency of severe crashes by hour of day
    hourly_severe = (
        collisions_clean[collisions_clean["SEVERE_CRASH"]]
        .groupby("CRASH HOUR")
        .size()
    )

    plt.figure(figsize=(10, 5))
    hourly_severe.plot(kind="bar", color="darkorange", edgecolor="black")
    plt.title("Frequency of Severe Collisions by Hour of the Day")
    plt.xlabel("Hour of the Day (0-23)")
    plt.ylabel("Number of Severe Crashes")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig("figure_3.png", dpi=300, bbox_inches="tight")
    print("Saved plot successfully to figure_3.png!")

    # Figure 4: seasonal distribution of severe crashes, by borough
    collisions_clean["CRASH MONTH"] = pd.to_datetime(
        collisions_clean["CRASH DATE"]
    ).dt.month
    seasonal_borough = (
        collisions_clean[
            (collisions_clean["SEVERE_CRASH"])
            & (collisions_clean["BOROUGH"] != "Unknown")
        ]
        .groupby(["CRASH MONTH", "BOROUGH"])
        .size()
        .unstack()
    )

    seasonal_borough.plot(kind="bar", figsize=(11, 6), edgecolor="black")
    plt.title("Seasonal Distribution of Severe Crashes by Borough")
    plt.xlabel("Month of the Year")
    plt.ylabel("Number of Severe Crashes")
    plt.legend(title="Borough")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig("figure_4.png", dpi=300, bbox_inches="tight")
    print("Saved plot successfully to figure_4.png!")

    # Figure 5: heatmap of severe-crash rate by borough & hour (RQ2)
    df_rq2 = collisions_clean[
        collisions_clean["BOROUGH"] != "Unknown"
    ].copy()

    heatmap_data = (
        df_rq2.groupby(["BOROUGH", "CRASH HOUR"])["SEVERE_CRASH"]
        .agg(severe_rate="mean", total_crashes="count")
        .reset_index()
    )
    pivot_table = heatmap_data.pivot(
        index="BOROUGH", columns="CRASH HOUR", values="severe_rate"
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pivot_table, cmap="YlOrRd", linewidths=0.5,
        cbar_kws={"label": "Severe-Crash Rate"},
    )
    plt.title("Severe-Crash Rate by Borough and Hour of the Day (RQ2)")
    plt.xlabel("Hour of the Day (0 - 23)")
    plt.ylabel("Borough")
    plt.savefig("figure_5.png", dpi=300, bbox_inches="tight")
    print("Saved plot successfully to figure_5.png!")


# ---------------------------------------------------------------------------
# 2. AIR QUALITY
# ---------------------------------------------------------------------------
def load_and_filter_air_quality(path="air_quality.csv"):
    """Load the air quality CSV and keep only the Ozone (O3) 'Summer
    mean' measure at Borough level.
    """
    air_quality = pd.read_csv(path)
    air_quality = air_quality.dropna(axis=1, how="all")

    filtered_air = air_quality[
        (air_quality["Name"] == "Ozone (O3)")
        & (air_quality["Measure"] == "Summer mean")
        & (air_quality["Geo Type Name"] == "Borough")
    ]

    columns_to_keep = ["Geo Join ID", "Time Period", "Data Value"]
    filtered_air = filtered_air[columns_to_keep]

    return filtered_air


def summarize_air_quality(filtered_air):
    """Check for missing data in the filtered air quality table."""
    total_rows, total_cols = filtered_air.shape
    print(f"For filter_air data -> total rows: {total_rows}, "
          f"total columns: {total_cols}"
        )
    print("=== MISSING DATA CHECK: filtered_air ===")
    missing_per_col = filtered_air.isna().sum()
    total_missing = missing_per_col.sum()

    print("Missing values per column:")
    print(missing_per_col)
    print(f"\nTotal missing values in filtered_air: {total_missing}")

    if total_missing == 0:
        print(
            "Conclusion: The filtered air quality dataset has NO "
            "missing data."
        )
    else:
        print(
            f"Conclusion: There are {total_missing} missing values in the "
            "filtered dataset."
        )


# ---------------------------------------------------------------------------
# 3. POVERTY
# ---------------------------------------------------------------------------
def load_poverty(path="neighborhood_poverty.csv"):
    """Load the neighborhood poverty CSV and add a Higher-Poverty /
    Lower-Poverty group column based on the median split.
    """
    poverty_df = pd.read_csv(path)
    median_val = poverty_df["Percent"].median()
    poverty_df["Poverty_group"] = poverty_df["Percent"].apply(
        lambda x: "Higher-Poverty" if x >= median_val else "Lower-Poverty"
    )
    return poverty_df


def summarize_poverty(poverty_df):
    """Check for missing data in the poverty table."""
    total_rows, total_cols = poverty_df.shape
    print(f"For poverty_df data -> total rows: {total_rows}, "
          f"total columns: {total_cols}")

    print("=== MISSING DATA CHECK: neighborhood_poverty ===")
    missing_per_col = poverty_df.isna().sum()
    total_missing = missing_per_col.sum()

    print("Missing values per column:")
    print(missing_per_col)
    print(f"\nTotal missing values in the entire dataset: {total_missing}")

    if total_missing == 0:
        print(
            "Conclusion: The neighborhood_poverty dataset has NO "
            "missing data."
        )
    else:
        print(
            f"Conclusion: There are {total_missing} missing values in the "
            "dataset that need to be addressed."
        )


# ---------------------------------------------------------------------------
# 4. JOIN RQ3: collisions + air quality
# ---------------------------------------------------------------------------
def prepare_collisions_for_join(collisions_clean):
    """Handle missing data tier-by-tier and prepare the join key
    (BOROUGH_ID, YEAR) on collisions_clean.
    """
    # Tier 1: high missingness (> 50%) -> drop these columns entirely
    cols_to_drop = [
        "VEHICLE TYPE CODE 5", "CONTRIBUTING FACTOR VEHICLE 5",
        "VEHICLE TYPE CODE 4", "CONTRIBUTING FACTOR VEHICLE 4",
        "VEHICLE TYPE CODE 3", "CONTRIBUTING FACTOR VEHICLE 3",
        "OFF STREET NAME",
    ]
    collisions_clean = collisions_clean.drop(columns=cols_to_drop)

    # Tier 2: moderate missingness (5% - 50%) -> fill with 'Unknown'
    cat_to_impute = [
        "CROSS STREET NAME", "ZIP CODE", "BOROUGH", "ON STREET NAME",
        "VEHICLE TYPE CODE 2", "CONTRIBUTING FACTOR VEHICLE 2", "LOCATION",
    ]
    for col in cat_to_impute:
        if col in collisions_clean.columns:
            collisions_clean[col] = collisions_clean[col].fillna("Unknown")

    # Coordinates (Latitude/Longitude) -> fill with median to keep them
    # roughly centered
    for coord_col in ["LATITUDE", "LONGITUDE"]:
        if coord_col in collisions_clean.columns:
            collisions_clean[coord_col] = collisions_clean[coord_col].fillna(
                collisions_clean[coord_col].median()
            )

    collisions_clean = collisions_clean[
        collisions_clean["BOROUGH"] != "Unknown"
    ]

    # Extract year from crash date, normalize dtypes for the join
    collisions_clean["YEAR"] = pd.to_datetime(
        collisions_clean["CRASH DATE"]
    ).dt.year.astype(str)

    collisions_clean["BOROUGH_ID"] = (
        collisions_clean["BOROUGH"].map(BOROUGH_MAPPING).astype(int)
    )

    return collisions_clean


def join_rq3(collisions_clean, filtered_air):
    """Join collisions with air quality on (BOROUGH_ID, YEAR)."""
    filtered_air = filtered_air.copy()
    filtered_air["Time Period"] = filtered_air["Time Period"].astype(str)
    filtered_air["Geo Join ID"] = filtered_air["Geo Join ID"].astype(int)

    joined_rq3 = pd.merge(
        collisions_clean,
        filtered_air,
        left_on=["BOROUGH_ID", "YEAR"],
        right_on=["Geo Join ID", "Time Period"],
        how="inner",
    )
    joined_rq3 = joined_rq3.rename(columns={"Data Value": "Ozone_Level"})

    print(f"Pipeline complete! Final joined dataset shape: {joined_rq3.shape}")
    return joined_rq3


def summarize_rq3(joined_rq3):
    """Summarize the joined RQ3 table."""
    cols_to_keep_rq3 = [
        "CRASH DATE", "CRASH TIME", "BOROUGH",
        "VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2",
        "CONTRIBUTING FACTOR VEHICLE 1", "CONTRIBUTING FACTOR VEHICLE 2",
        "NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED",
        "Ozone_Level", "SEVERE_CRASH",
    ]
    joined_rq3_model = joined_rq3[cols_to_keep_rq3].copy()

    print("=== SUMMARY OF VARIABLES OF INTEREST (joined_rq3_model)===")
    print("--- QUANTITATIVE VARIABLES (7-NUMBER SUMMARY) ---")
    quant_vars = [
        "Ozone_Level", "NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED"
    ]
    print(joined_rq3_model[quant_vars].describe(percentiles=[.25, .50, .75]))

    cat_vars = [
        "BOROUGH", "VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2",
        "CONTRIBUTING FACTOR VEHICLE 1", "CONTRIBUTING FACTOR VEHICLE 2",
    ]
    for col in cat_vars:
        print(f"\n--- CATEGORICAL SUMMARY: {col} ---")
        print(joined_rq3_model[col].value_counts().head(10))
    print(joined_rq3_model)
    return joined_rq3_model


def plot_rq3(joined_rq3_model):
    """Plots for RQ3 (only run when RUN_PLOTS is True)."""
    print("=== VISUALIZATION ===")

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="SEVERE_CRASH", y="Ozone_Level", data=joined_rq3_model,
        hue="SEVERE_CRASH", palette="Set2", legend=False,
    )
    plt.title("Distribution of Summer Ozone Levels by Crash Severity")
    plt.xlabel("Severe Crash (Injury or Fatality)")
    plt.ylabel("Summer Mean Ozone (ppb)")
    plt.show()

    borough_ozone = (
        joined_rq3_model.groupby("BOROUGH")["Ozone_Level"].mean().reset_index()
    )

    plt.figure(figsize=(9, 5))
    plt.bar(
        borough_ozone["BOROUGH"], borough_ozone["Ozone_Level"],
        color="#2b5c8f", edgecolor="black",
    )
    plt.title("Average Summer Ozone Levels by Borough (RQ3)")
    plt.xlabel("Borough")
    plt.ylabel("Summer Mean Ozone (ppb)")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("figure_6.png", dpi=300, bbox_inches="tight")
    print("Saved plot successfully to figure_6.png!")


# ---------------------------------------------------------------------------
# 5. JOIN RQ4: (collisions + air quality) + poverty
# ---------------------------------------------------------------------------
def expand_poverty_timeperiod(poverty_df):
    """poverty_df uses ~5-year rolling windows (e.g. '2007-11',
    '2019-23'). This function "expands" each time period into
    individual annual rows so it can be joined year-by-year with the
    collisions data.
    """
    poverty_df = poverty_df.copy()
    poverty_df.columns = [col.upper() for col in poverty_df.columns]

    expanded_rows = []
    for _, row in poverty_df.iterrows():
        time_period = str(row["TIMEPERIOD"])
        if "-" not in time_period:
            continue

        start_str, end_str = time_period.split("-")
        start_year = int(start_str)
        end_year = 2000 + int(end_str) if len(end_str) == 2 else int(end_str)

        for year in range(start_year, end_year + 1):
            new_row = row.copy()
            new_row["EXPANDED_YEAR"] = year
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)


def build_poverty_unique(expanded_poverty_df):
    """Group by (BOROID, EXPANDED_YEAR), taking the mean poverty
    percent and the first poverty group, returning a single row per
    (borough, year).
    """
    poverty_subset = expanded_poverty_df[
        ["BOROID", "EXPANDED_YEAR", "PERCENT", "POVERTY_GROUP"]
    ].copy()

    poverty_unique = (
        poverty_subset.groupby(["BOROID", "EXPANDED_YEAR"])
        .agg(
            PERCENT=("PERCENT", "mean"),
            POVERTY_GROUP=("POVERTY_GROUP", "first"),
        )
        .reset_index()
    )
    poverty_unique["EXPANDED_YEAR"] = (
        poverty_unique["EXPANDED_YEAR"].astype(int)
    )
    poverty_unique["BOROID"] = poverty_unique["BOROID"].astype(int)

    return poverty_subset, poverty_unique


def join_rq4(joined_rq3, poverty_unique):
    """Join the RQ3 result with the poverty table on
    (BOROUGH_ID, YEAR).
    """
    joined_rq3 = joined_rq3.copy()
    joined_rq3["YEAR"] = joined_rq3["YEAR"].astype(int)
    joined_rq3["BOROUGH_ID"] = joined_rq3["BOROUGH_ID"].astype(int)

    joined_rq4 = pd.merge(
        joined_rq3,
        poverty_unique,
        left_on=["BOROUGH_ID", "YEAR"],
        right_on=["BOROID", "EXPANDED_YEAR"],
        how="inner",
    )
    joined_rq4 = joined_rq4.drop(
        columns=["BOROID", "EXPANDED_YEAR"], errors="ignore"
    )

    has_percent = "PERCENT" in joined_rq4.columns
    has_poverty_percent = "POVERTY_PERCENT" in joined_rq4.columns
    if has_percent and not has_poverty_percent:
        joined_rq4["POVERTY_PERCENT"] = joined_rq4["PERCENT"]

    return joined_rq4


def build_rq4_clean(joined_rq4):
    """Keep only the columns needed for the RQ4 analysis."""
    cols_to_keep = [
        "CRASH DATE", "CRASH TIME", "BOROUGH", "ZIP CODE",
        "LATITUDE", "LONGITUDE", "NUMBER OF PERSONS INJURED",
        "NUMBER OF PERSONS KILLED", "YEAR", "BOROUGH_ID",
        "Ozone_Level", "POVERTY_PERCENT", "POVERTY_GROUP", "SEVERE_CRASH",
    ]
    actual_cols = [col for col in cols_to_keep if col in joined_rq4.columns]
    return joined_rq4[actual_cols].copy()


def summarize_rq4(joined_rq4_clean):
    """Print descriptive stats for the cleaned RQ4 table."""
    print("=== DATASET SIZE ===")
    rows, cols = joined_rq4_clean.shape
    print(f"The dataset contains {rows} rows and {cols} columns.")
    print(
        "Rows represent: A single police-reported motor vehicle collision "
        "event in NYC."
    )
    print(
        "Columns represent: Specific attributes of the crash (time, "
        "location, severity) combined with neighborhood ozone levels and "
        "poverty percentages.\n"
    )

    print("=== MISSING DATA ===")
    vars_of_interest = [
        "SEVERE_CRASH", "Ozone_Level", "POVERTY_PERCENT",
        "POVERTY_GROUP", "BOROUGH",
    ]
    missing_data = joined_rq4_clean[vars_of_interest].isnull().sum()
    total_missing = missing_data.sum()

    print("Missing values per variable of interest:")
    print(missing_data)
    print(f"\nTotal missing values in variables of interest: {total_missing}")

    if total_missing == 0:
        print(
            "Conclusion: There is NO missing data for our variables of "
            "interest. This is because rows with missing critical features "
            "were proactively dropped via .dropna() prior to performing an "
            "'inner' merge across datasets.\n"
        )
    else:
        print("Conclusion: Missing data exists.\n")

    print("=== SUMMARY OF VARIABLES OF INTEREST (joined_rq4_clean)===")
    print("\n--- Quantitative Variables (7-Number Summary) ---")
    quant_vars = ["Ozone_Level", "POVERTY_PERCENT"]
    print(joined_rq4_clean[quant_vars].describe())

    print("\n--- Categorical Variables (Unique Values & Counts) ---")
    for col in ["SEVERE_CRASH", "POVERTY_GROUP", "BOROUGH"]:
        print(f"\nVariable: {col}")
        print(joined_rq4_clean[col].value_counts(dropna=False))


def add_severe_crash_numeric(joined_rq4_clean):
    """Add a SEVERE_CRASH_NUM (0/1) column for plotting/probabilities."""
    if joined_rq4_clean["SEVERE_CRASH"].dtype == bool:
        joined_rq4_clean["SEVERE_CRASH_NUM"] = joined_rq4_clean[
            "SEVERE_CRASH"
        ].astype(int)
    else:
        joined_rq4_clean["SEVERE_CRASH_NUM"] = joined_rq4_clean[
            "SEVERE_CRASH"
        ].replace({"Yes": 1, "No": 0, True: 1, False: 0})
    return joined_rq4_clean


def plot_rq4(joined_rq4_clean):
    """Plots for RQ4 (only run when RUN_PLOTS is True)."""
    sns.set_theme(style="whitegrid", palette="muted")

    # Plot 1: baseline probability of a severe crash by poverty group
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=joined_rq4_clean,
        x="POVERTY_GROUP",
        y="SEVERE_CRASH_NUM",
        hue="POVERTY_GROUP",
        palette=["#e74c3c", "#3498db"],
        legend=False,
    )
    plt.title(
        "Baseline Probability of a Severe Crash by Poverty Group",
        fontsize=14, fontweight="bold",
    )
    plt.ylabel("Proportion of Severe Crashes (Probability)")
    plt.xlabel("Neighborhood Socioeconomic Status (Poverty Group)")
    plt.savefig("figure_7.png", dpi=300, bbox_inches="tight")
    print("Saved plot successfully to figure_7.png!")

    # Plot 2: moderation effect of ozone on severe-crash probability
    plot_sample = joined_rq4_clean.sample(n=50000, random_state=42)
    print("PLOTTING figure_8 ...")
    sns.lmplot(
        data=plot_sample,
        x="Ozone_Level",
        y="SEVERE_CRASH_NUM",
        hue="POVERTY_GROUP",
        palette=["#e74c3c", "#3498db"],
        logistic=True,
        y_jitter=0.02,
        scatter_kws={"alpha": 0.1, "s": 10},
        aspect=1.5,
    )
    plt.title(
        "Moderation Effect: Impact of Ozone on Crash Severity Probability",
        fontsize=15, fontweight="bold",
    )
    plt.xlabel("Ozone Level (Summer Mean)")
    plt.ylabel("Probability of Severe Crash (1.0 = Severe)")
    plt.tight_layout()
    plt.savefig("figure_8.png", dpi=300, bbox_inches="tight")
    print("Saved plot successfully to figure_8.png!")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    """Run the full pipeline: load -> clean -> join -> summarize -> test."""
    # ----- Collisions -----
    collisions = load_collisions()
    collisions_clean = clean_collisions(collisions)
    summarize_collisions(collisions, collisions_clean)
    if RUN_PLOTS:
        plot_collisions(collisions_clean)

    # ----- Air quality -----
    filtered_air = load_and_filter_air_quality()
    summarize_air_quality(filtered_air)

    # ----- Poverty -----
    poverty_df = load_poverty()
    summarize_poverty(poverty_df)

    # ----- RQ3: collisions + air quality -----
    collisions_clean = prepare_collisions_for_join(collisions_clean)
    joined_rq3 = join_rq3(collisions_clean, filtered_air)
    joined_rq3_model = summarize_rq3(joined_rq3)
    run_rq3_machine_learning(joined_rq3_model)
    if RUN_PLOTS:
        plot_rq3(joined_rq3_model)

    # ----- RQ4: (RQ3) + poverty -----
    expanded_poverty_df = expand_poverty_timeperiod(poverty_df)
    poverty_subset, poverty_unique = build_poverty_unique(expanded_poverty_df)

    joined_rq4 = join_rq4(joined_rq3, poverty_unique)
    joined_rq4_clean = build_rq4_clean(joined_rq4)
    summarize_rq4(joined_rq4_clean)
    joined_rq4_clean = add_severe_crash_numeric(joined_rq4_clean)
    if RUN_PLOTS:
        plot_rq4(joined_rq4_clean)

    # ----- Sanity checks -----
    test.run_sanity_checks(
        collisions_clean,
        filtered_air,
        poverty_unique,
        joined_rq3_model,
        joined_rq3,
        joined_rq4_clean,
    )


if __name__ == "__main__":
    main()
