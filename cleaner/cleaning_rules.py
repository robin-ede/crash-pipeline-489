"""
Cleaning rules for crash data.
Implements outcome-agnostic cleaning steps for ML-ready Gold layer.
"""
import logging
import polars as pl

logging.basicConfig(level=logging.INFO, format="[cleaner:cleaning_rules] %(message)s")


def drop_leakage_and_ids(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove leakage and high-cardinality ID columns.
    Keep crash_record_id (it's the primary key).
    """
    cols_to_drop = [
        "veh_vehicle_id_list_json",
        "ppl_person_id_list_json",
    ]

    existing_drops = [c for c in cols_to_drop if c in df.columns]

    if existing_drops:
        df = df.drop(existing_drops)

    return df


def standardize_booleans(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize all columns ending in _i to boolean (1/0/null).
    Maps: y/yes/true/1 → 1, n/no/false/0 → 0, else → null
    """
    bool_cols = [c for c in df.columns if c.endswith("_i")]

    if not bool_cols:
        return df

    for col in bool_cols:
        df = df.with_columns(
            pl.col(col)
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.strip_chars()
            .replace({"y": "1", "yes": "1", "true": "1", "t": "1", "n": "0", "no": "0", "false": "0", "f": "0"})
            .cast(pl.Int8, strict=False)
            .alias(col)
        )

    return df


def engineer_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parse crash_date and create engineered time features.
    Adds: year, month, day, hour, is_weekend, hour_bin
    """
    df = df.with_columns(
        pl.col("crash_date").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f").alias("crash_date")
    )

    df = df.with_columns([
        pl.col("crash_date").dt.year().alias("year"),
        pl.col("crash_date").dt.month().alias("month"),
        pl.col("crash_date").dt.day().alias("day"),
        pl.col("crash_date").dt.hour().alias("hour"),
        # Polars weekday: Monday=1, Sunday=7
        pl.col("crash_date").dt.weekday().alias("day_of_week"),
        # Weekend is Saturday(6) and Sunday(7)
        (pl.col("crash_date").dt.weekday().is_in([6, 7])).cast(pl.Int8).alias("is_weekend"),
    ])

    # Hour bins: night(0-6), morning(7-12), afternoon(13-18), evening(19-23)
    df = df.with_columns(
        pl.when(pl.col("hour").is_between(0, 6))
        .then(pl.lit("night"))
        .when(pl.col("hour").is_between(7, 12))
        .then(pl.lit("morning"))
        .when(pl.col("hour").is_between(13, 18))
        .then(pl.lit("afternoon"))
        .otherwise(pl.lit("evening"))
        .alias("hour_bin")
    )

    return df


def clean_location_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Validate and bin latitude/longitude coordinates.
    Chicago bounds: ~41.6-42.1, -87.9--87.5
    Creates: lat_bin, lng_bin, grid_id
    """
    before_count = df.height
    df = df.filter(
        (pl.col("latitude").is_not_null()) &
        (pl.col("longitude").is_not_null()) &
        (pl.col("latitude") != 0.0) &
        (pl.col("longitude") != 0.0) &
        (pl.col("latitude").is_between(41.6, 42.1)) &
        (pl.col("longitude").is_between(-87.9, -87.5))
    )
    after_count = df.height
    dropped = before_count - after_count
    if dropped > 0:
        logging.info(f"Dropped {dropped} rows with invalid coordinates")

    df = df.with_columns([
        pl.col("latitude").round(2).alias("lat_bin"),
        pl.col("longitude").round(2).alias("lng_bin"),
    ])

    df = df.with_columns(
        (pl.col("lat_bin").cast(pl.Utf8) + "_" + pl.col("lng_bin").cast(pl.Utf8)).alias("grid_id")
    )

    return df


def process_vehicles_people_json(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parse JSON list columns for vehicles and people.
    Extract:
    - veh_truck_i, veh_mc_i (vehicle type flags)
    - ppl_age_mean, ppl_age_min, ppl_age_max (age statistics)
    """
    if "veh_vehicle_type_list_json" in df.columns:
        df = df.with_columns([
            pl.col("veh_vehicle_type_list_json")
            .str.to_uppercase()
            .str.contains("TRUCK")
            .fill_null(False)
            .cast(pl.Int8)
            .alias("veh_truck_i"),

            pl.col("veh_vehicle_type_list_json")
            .str.to_uppercase()
            .str.contains("MOTORCYCLE|MC")
            .fill_null(False)
            .cast(pl.Int8)
            .alias("veh_mc_i"),
        ])

    if "ppl_age_list_json" in df.columns:
        df = df.with_columns(
            pl.col("ppl_age_list_json")
            .str.replace_all(r'[\[\]\"]', '')
            .str.split(",")
            .list.eval(pl.element().cast(pl.Int64, strict=False))
            .list.eval(pl.element().filter((pl.element() >= 0) & (pl.element() <= 110)))
            .alias("ppl_ages_parsed")
        )

        df = df.with_columns([
            pl.col("ppl_ages_parsed").list.mean().alias("ppl_age_mean"),
            pl.col("ppl_ages_parsed").list.min().alias("ppl_age_min"),
            pl.col("ppl_ages_parsed").list.max().alias("ppl_age_max"),
        ])

        df = df.drop("ppl_ages_parsed")

    json_cols_to_drop = [c for c in df.columns if c.endswith("_json")]
    if json_cols_to_drop:
        df = df.drop(json_cols_to_drop)

    return df


def standardize_categoricals(df: pl.DataFrame, nulls_fixed_counter=None) -> pl.DataFrame:
    """
    Clean categorical text fields:
    - Trim whitespace, lowercase
    - Map UNKNOWN/UNK/N/A to null

    Args:
        df: DataFrame to clean
        nulls_fixed_counter: Optional Prometheus Counter to track null replacements
    """
    cat_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    total_nulls_fixed = 0

    for col in cat_cols:
        # Count nulls before cleaning
        before_nulls = df[col].null_count()

        df = df.with_columns(
            pl.col(col)
            .str.strip_chars()
            .str.to_lowercase()
            .replace(["unknown", "unk", "n/a", "na", ""], None)
            .alias(col)
        )

        # Count nulls after cleaning
        after_nulls = df[col].null_count()
        nulls_fixed = after_nulls - before_nulls
        total_nulls_fixed += nulls_fixed

    # Update Prometheus metric if provided
    if nulls_fixed_counter and total_nulls_fixed > 0:
        nulls_fixed_counter.inc(total_nulls_fixed)
        logging.info(f"Fixed {total_nulls_fixed} null/unknown values across {len(cat_cols)} categorical columns")

    return df


def handle_outliers(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cap extreme values for counts to handle outliers.
    """
    if "veh_count" in df.columns:
        df = df.with_columns(
            pl.col("veh_count")
            .cast(pl.Int32, strict=False)
            .alias("veh_count")
        )
        df = df.with_columns(
            pl.when(pl.col("veh_count") > 5)
            .then(5)
            .otherwise(pl.col("veh_count"))
            .alias("veh_count")
        )

    if "ppl_count" in df.columns:
        df = df.with_columns(
            pl.col("ppl_count")
            .cast(pl.Int32, strict=False)
            .alias("ppl_count")
        )
        df = df.with_columns(
            pl.when(pl.col("ppl_count") > 10)
            .then(10)
            .otherwise(pl.col("ppl_count"))
            .alias("ppl_count")
        )

    if "ppl_age_max" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("ppl_age_max") > 110)
            .then(110)
            .otherwise(pl.col("ppl_age_max"))
            .alias("ppl_age_max")
        )

    return df


def deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove duplicate crash_record_id rows (keep first).
    """
    before = df.height
    df = df.unique(subset=["crash_record_id"], keep="first", maintain_order=True)
    after = df.height

    if before > after:
        logging.info(f"Removed {before - after} duplicate crash_record_id rows")

    return df


def create_injury_target(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create binary 'injury' target variable for any injury outcome.
    injury = 1 if injuries_total >= 1, else 0

    This outcome identifies crashes with at least one injury of any severity.
    Uses injuries_total as the primary indicator, with fallback to summing
    individual injury categories if injuries_total is unavailable.

    NOTE: This function DROPS all injury columns after creating the target
    to prevent label leakage.
    """
    injury_cols = [
        "injuries_total",
        "injuries_fatal",
        "injuries_incapacitating",
        "injuries_non_incapacitating",
        "injuries_reported_not_evident"
    ]

    available_cols = [c for c in injury_cols if c in df.columns]

    if not available_cols:
        logging.warning(f"Cannot create 'injury' target: no injury columns available")
        return df

    # Prefer injuries_total if available, otherwise sum individual categories
    if "injuries_total" in df.columns:
        df = df.with_columns(
            (pl.col("injuries_total").fill_null(0) >= 1)
            .cast(pl.Int8)
            .alias("injury")
        )
        logging.info("Created 'injury' binary target using injuries_total")
    else:
        # Fallback: sum all available injury categories
        injury_sum = pl.lit(0)
        for col in available_cols:
            injury_sum = injury_sum + pl.col(col).fill_null(0)

        df = df.with_columns(
            (injury_sum >= 1)
            .cast(pl.Int8)
            .alias("injury")
        )
        logging.info(f"Created 'injury' binary target by summing {len(available_cols)} injury columns")

    # Drop all injury columns to prevent label leakage
    cols_to_drop = [c for c in injury_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)
        logging.info(f"Dropped {len(cols_to_drop)} injury columns to prevent leakage")

    return df


def apply_cleaning_rules(df: pl.DataFrame, nulls_fixed_counter=None) -> pl.DataFrame:
    """
    Main cleaning orchestrator.
    Applies all cleaning steps in order to produce ML-ready Gold data.

    Args:
        df: Raw Silver DataFrame
        nulls_fixed_counter: Optional Prometheus Counter to track null replacements

    Returns:
        Cleaned DataFrame ready for Gold layer
    """
    if df.is_empty():
        logging.warning("Empty DataFrame received for cleaning")
        return df

    if "crash_record_id" not in df.columns:
        raise ValueError("crash_record_id column missing from Silver data")

    logging.info(f"Starting cleaning: {df.height} rows, {df.width} columns")

    df = drop_leakage_and_ids(df)
    df = standardize_booleans(df)
    df = engineer_time_features(df)
    df = clean_location_features(df)
    df = process_vehicles_people_json(df)
    df = standardize_categoricals(df, nulls_fixed_counter)
    df = handle_outliers(df)
    df = deduplicate(df)
    df = create_injury_target(df)

    logging.info(f"Cleaning complete: {df.height} rows, {df.width} columns")

    return df
