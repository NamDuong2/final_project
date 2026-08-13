"""
Author: Nam Duong
CSE 163 AA

This is test file
"""

def run_sanity_checks(
    collisions_clean,
    filtered_air,
    poverty_unique,
    joined_rq3_model,
    joined_rq3,
    joined_rq4_clean,
):
    """Run all the asserts that validate the pipeline's correctness."""
    print("\n" + "=" * 60)
    print("EXECUTING DEEP-DIVE TESTS FOR EACH DATASET")
    print("=" * 60)

    # 1. Collisions
    print("Testing Collisions...")
    assert "Unknown" not in collisions_clean["BOROUGH"].values, (
        "Test Failed: 'Unknown' boroughs were not filtered out!"
    )
    assert collisions_clean[
        ["VEHICLE TYPE CODE 1", "CONTRIBUTING FACTOR VEHICLE 1", "CRASH DATE"]
    ].isna().sum().sum() == 0, "Test Failed: Critical columns contain NaNs!"
    assert "VEHICLE TYPE CODE 5" not in collisions_clean.columns, (
        "Test Failed: Tier 1 high-missingness columns were not dropped!"
    )
    assert collisions_clean["LATITUDE"].isna().sum() == 0, (
        "Test Failed: Latitude imputation failed!"
    )
    assert collisions_clean["CRASH HOUR"].between(0, 23).all(), (
        "Test Failed: CRASH HOUR out of bounds!"
    )

    # 2. Air quality
    print("Testing Air Quality...")
    assert list(filtered_air.columns) == [
        "Geo Join ID", "Time Period", "Data Value"
    ], "Test Failed: Unwanted columns were not dropped!"
    assert filtered_air["Time Period"].astype(int).between(2000, 2026).all(), (
        "Test Failed: Time Period out of expected bounds!"
    )
    assert filtered_air["Data Value"].min() >= 0, (
        "Test Failed: Ozone level cannot be negative!"
    )

    # 3. Poverty
    print("Testing Poverty...")
    assert poverty_unique["PERCENT"].between(0, 100).all(), (
        "Test Failed: Poverty percent out of 0-100 range!"
    )
    assert set(poverty_unique["POVERTY_GROUP"].unique()).issubset(
        {"Higher-Poverty", "Lower-Poverty"}
    ), "Test Failed: Invalid POVERTY_GROUP categories!"
    assert poverty_unique["EXPANDED_YEAR"].dtype == int, (
        "Test Failed: EXPANDED_YEAR is not integer type!"
    )

    # 4. Joined datasets (RQ3 & RQ4)
    print("Testing Joined Datasets...")
    assert len(joined_rq3_model) <= len(collisions_clean), (
        "Test Failed: RQ3 join created duplicate rows (Cartesian explosion)!"
    )
    assert len(joined_rq4_clean) <= len(joined_rq3), (
        "Test Failed: RQ4 join created duplicate rows!"
    )
    assert "POVERTY_PERCENT" in joined_rq4_clean.columns, (
        "Test Failed: POVERTY_PERCENT mapping failed in RQ4!"
    )

    print("ALL DEEP-DIVE DATASET TESTS PASSED!")

