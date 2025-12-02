"""
DuckDB writer with schema creation and upsert logic.
Ensures non-redundancy via PRIMARY KEY constraint.
"""
import logging
import time
import duckdb
import polars as pl
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[cleaner:duckdb_writer] %(message)s")


def ensure_schema_and_table(con: duckdb.DuckDBPyConnection, table_name: str):
    """
    Create schema and table if they don't exist.
    Table has PRIMARY KEY on crash_record_id for non-redundancy.

    Args:
        con: DuckDB connection
        table_name: Fully qualified table name (e.g., "gold_crashes" or "main.crashes")
    """
    # For simplicity, if table_name contains a dot, treat first part as schema
    # Otherwise use main schema
    if "." in table_name:
        schema_name, base_table = table_name.split(".", 1)
    else:
        schema_name = "main"
        base_table = table_name

    # Get the actual catalog name from DuckDB
    catalog_result = con.execute("SELECT current_database()").fetchone()
    catalog_name = catalog_result[0] if catalog_result else None

    # Create schema if not main
    if schema_name != "main":
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')

    # Use explicit 3-part catalog reference if catalog name is available
    # Otherwise fall back to 2-part reference
    if catalog_name:
        full_table_name = f'"{catalog_name}"."{schema_name}"."{base_table}"'
    else:
        full_table_name = f'"{schema_name}"."{base_table}"'

    # Create table with PRIMARY KEY and all engineered columns
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {full_table_name} (
            crash_record_id VARCHAR PRIMARY KEY,
            crash_date TIMESTAMP,

            -- Original crash attributes
            posted_speed_limit INTEGER,
            traffic_control_device VARCHAR,
            device_condition VARCHAR,
            weather_condition VARCHAR,
            lighting_condition VARCHAR,
            first_crash_type VARCHAR,
            trafficway_type VARCHAR,
            alignment VARCHAR,
            roadway_surface_cond VARCHAR,
            road_defect VARCHAR,
            num_units INTEGER,
            crash_hour INTEGER,
            crash_day_of_week INTEGER,
            crash_month INTEGER,
            beat_of_occurrence INTEGER,

            -- Boolean flags (standardized)
            intersection_related_i TINYINT,
            hit_and_run_i TINYINT,

            -- Location (original + binned)
            latitude DOUBLE,
            longitude DOUBLE,
            lat_bin DOUBLE,
            lng_bin DOUBLE,
            grid_id VARCHAR,

            -- Aggregated vehicle/people counts
            veh_count INTEGER,
            ppl_count INTEGER,

            -- Engineered time features
            year INTEGER,
            month TINYINT,
            day TINYINT,
            day_of_week TINYINT,
            hour TINYINT,
            is_weekend TINYINT,
            hour_bin VARCHAR,

            -- Vehicle flags
            veh_truck_i TINYINT,
            veh_mc_i TINYINT,

            -- People age statistics
            ppl_age_mean DOUBLE,
            ppl_age_min INTEGER,
            ppl_age_max INTEGER,

            -- Target variable for Any Injury outcome
            injury TINYINT
        )
    """)

    # Return the properly formatted table name for use in other functions
    return full_table_name


def upsert_to_gold(con: duckdb.DuckDBPyConnection, table_name: str, df: pl.DataFrame) -> dict:
    """
    Insert new rows into Gold table, or update existing ones (true upsert).

    Args:
        con: DuckDB connection
        table_name: Target table name
        df: Cleaned DataFrame to insert/update

    Returns:
        dict with counts: {"inserted": int, "updated": int}
    """
    if df.is_empty():
        logging.warning("Empty DataFrame, nothing to upsert")
        return {"inserted": 0, "updated": 0}

    # Get row count before and identify existing records
    count_before = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    # Check how many records from incoming data already exist
    con.register("temp_check", df)
    existing_count = con.execute(f"""
        SELECT COUNT(*)
        FROM {table_name} t
        WHERE t.crash_record_id IN (SELECT crash_record_id FROM temp_check)
    """).fetchone()[0]
    con.unregister("temp_check")

    # Convert crash_date to string format for DuckDB compatibility
    if "crash_date" in df.columns:
        df = df.with_columns(
            pl.col("crash_date").dt.strftime("%Y-%m-%d %H:%M:%S").alias("crash_date")
        )

    # Register DataFrame as temp view
    con.register("temp_cleaned", df)

    # Insert with ON CONFLICT DO UPDATE SET (upsert: update existing records)
    # Cast crash_date string back to TIMESTAMP in DuckDB
    con.execute(f"""
        INSERT INTO {table_name}
        SELECT
            crash_record_id,
            CAST(crash_date AS TIMESTAMP) as crash_date,
            posted_speed_limit,
            traffic_control_device,
            device_condition,
            weather_condition,
            lighting_condition,
            first_crash_type,
            trafficway_type,
            alignment,
            roadway_surface_cond,
            road_defect,
            num_units,
            crash_hour,
            crash_day_of_week,
            crash_month,
            beat_of_occurrence,
            intersection_related_i,
            hit_and_run_i,
            latitude,
            longitude,
            lat_bin,
            lng_bin,
            grid_id,
            veh_count,
            ppl_count,
            year,
            month,
            day,
            day_of_week,
            hour,
            is_weekend,
            hour_bin,
            veh_truck_i,
            veh_mc_i,
            ppl_age_mean,
            ppl_age_min,
            ppl_age_max,
            injury
        FROM temp_cleaned
        ON CONFLICT (crash_record_id) DO UPDATE SET
            crash_date = EXCLUDED.crash_date,
            posted_speed_limit = EXCLUDED.posted_speed_limit,
            traffic_control_device = EXCLUDED.traffic_control_device,
            device_condition = EXCLUDED.device_condition,
            weather_condition = EXCLUDED.weather_condition,
            lighting_condition = EXCLUDED.lighting_condition,
            first_crash_type = EXCLUDED.first_crash_type,
            trafficway_type = EXCLUDED.trafficway_type,
            alignment = EXCLUDED.alignment,
            roadway_surface_cond = EXCLUDED.roadway_surface_cond,
            road_defect = EXCLUDED.road_defect,
            num_units = EXCLUDED.num_units,
            crash_hour = EXCLUDED.crash_hour,
            crash_day_of_week = EXCLUDED.crash_day_of_week,
            crash_month = EXCLUDED.crash_month,
            beat_of_occurrence = EXCLUDED.beat_of_occurrence,
            intersection_related_i = EXCLUDED.intersection_related_i,
            hit_and_run_i = EXCLUDED.hit_and_run_i,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            lat_bin = EXCLUDED.lat_bin,
            lng_bin = EXCLUDED.lng_bin,
            grid_id = EXCLUDED.grid_id,
            veh_count = EXCLUDED.veh_count,
            ppl_count = EXCLUDED.ppl_count,
            year = EXCLUDED.year,
            month = EXCLUDED.month,
            day = EXCLUDED.day,
            day_of_week = EXCLUDED.day_of_week,
            hour = EXCLUDED.hour,
            is_weekend = EXCLUDED.is_weekend,
            hour_bin = EXCLUDED.hour_bin,
            veh_truck_i = EXCLUDED.veh_truck_i,
            veh_mc_i = EXCLUDED.veh_mc_i,
            ppl_age_mean = EXCLUDED.ppl_age_mean,
            ppl_age_min = EXCLUDED.ppl_age_min,
            ppl_age_max = EXCLUDED.ppl_age_max,
            injury = EXCLUDED.injury
    """)

    # Get row count after
    count_after = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    inserted = count_after - count_before
    updated = existing_count

    logging.info(f"Upsert complete: {inserted} inserted, {updated} updated")

    # Unregister temp view
    con.unregister("temp_cleaned")

    return {"inserted": inserted, "updated": updated}


def verify_gold_table(con: duckdb.DuckDBPyConnection, table_name: str):
    """
    Run post-write verification checks.

    Args:
        con: DuckDB connection
        table_name: Table to verify
    """
    # Check for duplicate crash_record_ids
    dups = con.execute(f"""
        SELECT crash_record_id, COUNT(*) as cnt
        FROM {table_name}
        GROUP BY crash_record_id
        HAVING COUNT(*) > 1
    """).fetchall()

    if dups:
        logging.error(f"Found {len(dups)} duplicate crash_record_id values!")
        for crash_id, cnt in dups[:5]:
            logging.error(f"  {crash_id}: {cnt} occurrences")

    # Count total rows
    total = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logging.info(f"Total rows in table: {total}")


def write_to_gold(db_path: str, table_name: str, df: pl.DataFrame) -> dict:
    """
    Main entry point: ensure schema, upsert data, verify.

    Args:
        db_path: Path to DuckDB file
        table_name: Fully qualified table name
        df: Cleaned DataFrame

    Returns:
        dict with upsert counts and total_rows
    """
    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Retry logic for handling concurrent read connections
    max_retries = 5
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            con = duckdb.connect(db_path)
            try:
                # Get properly quoted table name
                full_table_name = ensure_schema_and_table(con, table_name)
                result = upsert_to_gold(con, full_table_name, df)
                verify_gold_table(con, full_table_name)

                # Add total row count for metrics
                total_rows = con.execute(f"SELECT COUNT(*) FROM {full_table_name}").fetchone()[0]
                result['total_rows'] = total_rows

                return result
            finally:
                con.close()
        except duckdb.IOException as e:
            if "Could not set lock" in str(e) and attempt < max_retries - 1:
                logging.warning(f"Database locked, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                raise
