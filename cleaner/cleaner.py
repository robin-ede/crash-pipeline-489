"""
Cleaner service: RabbitMQ consumer that processes Silver CSV → Gold DuckDB.
Listens to 'clean' queue, downloads Silver CSV from MinIO, applies cleaning rules,
and writes to DuckDB with idempotent upsert logic.
"""
import os
import json
import logging
import time
import socket
import traceback
from typing import Optional

import pika
from pika.exceptions import AMQPConnectionError, ProbableAccessDeniedError, ProbableAuthenticationError
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from minio_io import minio_client, read_csv_from_minio
from cleaning_rules import apply_cleaning_rules
from duckdb_writer import write_to_gold

# ---------------------------------
# Logging
# ---------------------------------
logging.basicConfig(level=logging.INFO, format="[cleaner] %(message)s")
logging.getLogger("pika").setLevel(logging.WARNING)

# ---------------------------------
# Prometheus Metrics
# ---------------------------------
messages_processed = Counter('cleaner_messages_processed_total', 'Total clean messages processed')
messages_failed = Counter('cleaner_messages_failed_total', 'Total clean messages that failed')
rows_cleaned = Counter('cleaner_rows_cleaned_total', 'Total rows cleaned and written to Gold')
rows_inserted = Counter('cleaner_rows_inserted_total', 'Total rows inserted into Gold')
rows_updated = Counter('cleaner_rows_updated_total', 'Total rows updated in Gold')
nulls_fixed = Counter('cleaner_nulls_fixed_total', 'Total null values fixed during cleaning')
processing_duration = Histogram('cleaner_processing_duration_seconds', 'Time spent processing clean jobs', buckets=[0.5, 1, 2, 5, 10, 30, 60, 120])
duckdb_write_duration = Histogram('cleaner_duckdb_write_duration_seconds', 'Time spent writing to DuckDB', buckets=[0.1, 0.5, 1, 2, 5, 10, 30])
current_job_status = Gauge('cleaner_current_job_status', 'Current cleaning job status (1=processing, 0=idle)')
gold_table_row_count = Gauge('cleaner_gold_table_row_count', 'Current row count in Gold table')
gold_db_file_size_bytes = Gauge('cleaner_gold_db_file_size_bytes', 'Gold DuckDB file size in bytes')

# ---------------------------------
# Environment Configuration
# ---------------------------------
RABBIT_URL = os.getenv("RABBITMQ_URL")
CLEAN_QUEUE = os.getenv("CLEAN_QUEUE", "clean")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS = os.getenv("MINIO_USER")
MINIO_SECRET = os.getenv("MINIO_PASS")
MINIO_SECURE = os.getenv("MINIO_SSL", "false").lower() in ("1", "true", "t", "yes", "y")


def wait_for_port(host: str, port: int, tries: int = 60, delay: float = 1.0) -> bool:
    """Wait for TCP port to become available."""
    for _ in range(tries):
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except OSError:
            time.sleep(delay)
    return False


def initialize_metrics():
    """
    Initialize metrics with current database state on startup.
    This ensures the gold_table_row_count gauge reflects the actual row count
    even after service restarts.
    """
    try:
        import duckdb
        GOLD_DB_PATH = os.getenv("GOLD_DB_PATH", "/data/gold/gold.duckdb")

        print(f"[cleaner] Checking for Gold DB at: {GOLD_DB_PATH}")

        if os.path.exists(GOLD_DB_PATH):
            print(f"[cleaner] Gold DB exists, querying row count and file size...")

            # Get file size
            file_size = os.path.getsize(GOLD_DB_PATH)
            gold_db_file_size_bytes.set(file_size)
            print(f"[cleaner] ✓ Initialized gold_db_file_size_bytes metric: {file_size} bytes")

            # Get row count
            con = duckdb.connect(GOLD_DB_PATH, read_only=True)
            try:
                result = con.execute('SELECT COUNT(*) FROM "gold"."main"."crashes"').fetchone()
                if result:
                    current_count = result[0]
                    gold_table_row_count.set(current_count)
                    print(f"[cleaner] ✓ Initialized gold_table_row_count metric: {current_count} rows")
                    logging.info(f"Initialized gold_table_row_count and file_size metrics: {current_count} rows, {file_size} bytes")
            except Exception as e:
                print(f"[cleaner] ⚠ Could not query Gold table for row count: {e}")
                logging.warning(f"Could not query Gold table for row count: {e}")
            finally:
                con.close()
        else:
            print(f"[cleaner] Gold database not found at {GOLD_DB_PATH}, skipping metric initialization")
            logging.info(f"Gold database not found at {GOLD_DB_PATH}, skipping metric initialization")
    except Exception as e:
        print(f"[cleaner] ⚠ Could not initialize gold_table_row_count metric: {e}")
        import traceback
        traceback.print_exc()
        logging.warning(f"Could not initialize gold_table_row_count metric: {e}")


def run_clean_job(msg: dict):
    """
    Process a clean job:
    1. Locate Silver CSV in MinIO
    2. Download and load into DataFrame
    3. Apply cleaning rules
    4. Write to Gold DuckDB with upsert logic

    Args:
        msg: RabbitMQ message with keys:
            - corr_id: Correlation ID
            - xform_bucket: Transform bucket name
            - prefix: Dataset prefix (e.g., "crash")
            - gold_db_path: Path to DuckDB file
            - gold_table: Target table name
    """
    corr_id = msg.get("corr_id")
    xform_bucket = msg.get("xform_bucket")
    prefix = msg.get("prefix", "crash")
    gold_db_path = msg.get("gold_db_path", "/data/gold/gold.duckdb")
    gold_table = msg.get("gold_table", "main.crashes")

    if not corr_id or not xform_bucket:
        raise ValueError("Missing required fields: corr_id, xform_bucket")

    logging.info(f"Processing clean job for corr_id={corr_id}")

    # Construct Silver CSV path
    silver_key = f"{prefix}/corr={corr_id}/merged.csv"

    # Create MinIO client
    cli = minio_client(MINIO_ENDPOINT, MINIO_ACCESS, MINIO_SECRET, MINIO_SECURE)

    # Download Silver CSV
    try:
        silver_df = read_csv_from_minio(cli, xform_bucket, silver_key)
    except Exception as e:
        logging.error(f"Failed to read Silver CSV from s3://{xform_bucket}/{silver_key}: {e}")
        raise

    logging.info(f"Loaded Silver CSV: {silver_df.height} rows, {silver_df.width} columns")

    # Apply cleaning rules (pass nulls_fixed counter for tracking)
    cleaned_df = apply_cleaning_rules(silver_df, nulls_fixed_counter=nulls_fixed)

    # Track cleaned rows
    rows_cleaned.inc(cleaned_df.height)

    # Write to Gold DuckDB with timing
    write_start = time.time()
    result = write_to_gold(gold_db_path, gold_table, cleaned_df)
    duckdb_write_duration.observe(time.time() - write_start)

    # Track insert/update counts
    rows_inserted.inc(result['inserted'])
    rows_updated.inc(result['updated'])

    # Update gold table row count gauge
    if 'total_rows' in result:
        gold_table_row_count.set(result['total_rows'])

    # Update DuckDB file size gauge
    if os.path.exists(gold_db_path):
        file_size = os.path.getsize(gold_db_path)
        gold_db_file_size_bytes.set(file_size)

    logging.info(
        f"✓ Clean job complete for corr_id={corr_id}: "
        f"{result['inserted']} inserted, {result['updated']} updated"
    )


def start_consumer():
    """
    Start RabbitMQ consumer listening to clean queue.
    Processes messages and writes to Gold DuckDB.
    """
    # Start Prometheus metrics HTTP server
    start_http_server(8002)
    logging.info("Prometheus metrics server started on :8002/metrics")

    # Initialize metrics with current database state
    initialize_metrics()

    params = pika.URLParameters(RABBIT_URL)

    # Wait for RabbitMQ to be reachable
    host = params.host or "rabbitmq"
    port = params.port or 5672
    if not wait_for_port(host, port, tries=60, delay=1.0):
        raise SystemExit(f"[cleaner] RabbitMQ not reachable at {host}:{port}")

    # Connect to RabbitMQ with retries
    max_tries = 60
    base_delay = 1.5
    conn = None

    for i in range(1, max_tries + 1):
        try:
            conn = pika.BlockingConnection(params)
            break
        except (AMQPConnectionError, ProbableAccessDeniedError, ProbableAuthenticationError) as e:
            if i == 1:
                logging.info(f"Waiting for RabbitMQ @ {RABBIT_URL} …")
            if i % 10 == 0:
                logging.info(f"Still waiting (attempt {i}/{max_tries}): {e.__class__.__name__}")
            time.sleep(base_delay)

    if conn is None or not conn.is_open:
        raise SystemExit("[cleaner] Could not connect to RabbitMQ")

    ch = conn.channel()
    ch.queue_declare(queue=CLEAN_QUEUE, durable=True)
    ch.basic_qos(prefetch_count=1)

    def on_message(chx, method, props, body):
        start_time = time.time()
        current_job_status.set(1)  # Mark job as in progress
        try:
            msg = json.loads(body.decode("utf-8"))
            msg_type = msg.get("type", "")

            if msg_type != "clean":
                logging.info(f"Ignoring message type={msg_type!r}")
                chx.basic_ack(delivery_tag=method.delivery_tag)
                current_job_status.set(0)
                return

            logging.info(f"Received clean job: corr_id={msg.get('corr_id')}")
            run_clean_job(msg)
            chx.basic_ack(delivery_tag=method.delivery_tag)
            messages_processed.inc()
            processing_duration.observe(time.time() - start_time)

        except Exception:
            traceback.print_exc()
            messages_failed.inc()
            chx.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        finally:
            current_job_status.set(0)  # Mark job as completed

    logging.info(f"Up. Waiting for jobs on queue '{CLEAN_QUEUE}'")
    ch.basic_consume(queue=CLEAN_QUEUE, on_message_callback=on_message)

    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        try:
            ch.stop_consuming()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    start_consumer()
