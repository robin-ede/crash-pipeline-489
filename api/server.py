"""
FastAPI Backend Server for Chicago Crash ETL Dashboard
Provides REST API endpoints for health checks, data fetching, management, and reports.
"""
import os
import json
import logging
import subprocess
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import uuid
import atexit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pika
import duckdb
from minio import Minio
from minio.error import S3Error
import requests
import pandas as pd
import joblib
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time

# Import scheduler and report generator modules
import scheduler as sched_module
import report_generator

# =============================
# Configuration
# =============================
logging.basicConfig(level=logging.INFO, format="[api] %(message)s")
logger = logging.getLogger(__name__)

# =============================
# Prometheus Metrics
# =============================
http_requests_total = Counter('api_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
http_request_duration = Histogram('api_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
jobs_published = Counter('api_jobs_published_total', 'Total jobs published to queues', ['job_type'])
db_query_duration = Histogram('api_db_query_duration_seconds', 'Database query duration', buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5])
model_predictions = Counter('api_model_predictions_total', 'Total model predictions made')
prediction_latency = Histogram('api_prediction_latency_seconds', 'Model prediction latency', buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])
active_connections = Gauge('api_active_connections', 'Number of active connections')

# Environment variables
RABBIT_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
EXTRACT_QUEUE = os.getenv("EXTRACT_QUEUE", "extract")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER = os.getenv("MINIO_USER", "admin")
MINIO_PASS = os.getenv("MINIO_PASS", "admin123")
MINIO_SSL = os.getenv("MINIO_SSL", "false").lower() in ("1", "true", "yes")
SOCRATA_BASE = os.getenv("SOCRATA_BASE", "https://data.cityofchicago.org")
GOLD_DB_PATH = os.getenv("GOLD_DB_PATH", "/data/gold/gold.duckdb")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/artifacts/pipeline_calibrated.pkl")

# Dataset IDs
CRASHES_ID = "85ca-t3if"
VEHICLES_ID = "68nd-jvt3"
PEOPLE_ID = "u6pd-qa9d"

app = FastAPI(title="Chicago Crash ETL API", version="1.0.0")

# Global model cache
_model_cache = None

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics middleware to track all requests
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    # Skip metrics for the /metrics endpoint itself
    if request.url.path == "/metrics":
        return await call_next(request)

    active_connections.inc()
    start_time = time.time()
    endpoint = request.url.path
    method = request.method

    try:
        response = await call_next(request)
        status = response.status_code
        http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        http_request_duration.labels(method=method, endpoint=endpoint).observe(time.time() - start_time)
        return response
    except Exception as e:
        http_requests_total.labels(method=method, endpoint=endpoint, status=500).inc()
        raise
    finally:
        active_connections.dec()

# =============================
# Prometheus Metrics Endpoint
# =============================
@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# =============================
# Startup/Shutdown Events
# =============================
@app.on_event("startup")
async def startup_event():
    """Initialize scheduler and ensure buckets exist on startup."""
    logger.info("Initializing scheduler...")
    sched_module.init_scheduler(publish_job_from_config)

    # Ensure critical buckets exist
    logger.info("Checking MinIO buckets...")
    try:
        client = get_minio_client()
        required_buckets = ["raw-data", "transform-data"]
        for bucket in required_buckets:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                logger.info(f"Created missing bucket: {bucket}")
            else:
                logger.info(f"Bucket exists: {bucket}")
    except Exception as e:
        logger.error(f"Failed to ensure buckets exist: {e}")

    # Load ML model
    logger.info("Loading ML model...")
    try:
        load_model()
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Model endpoints will not be available")

    logger.info("API server started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down scheduler...")
    sched_module.shutdown_scheduler()
    logger.info("API server stopped")

# =============================
# Request/Response Models
# =============================
class FetchJobRequest(BaseModel):
    mode: str  # "streaming" or "backfill"
    since_days: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    start_time: Optional[str] = "00:00:00"
    end_time: Optional[str] = "23:59:59"
    crashes_columns: Optional[List[str]] = None
    vehicles_columns: Optional[List[str]] = None
    people_columns: Optional[List[str]] = None
    include_vehicles: bool = True
    include_people: bool = True

class MinioDeleteRequest(BaseModel):
    bucket: str
    prefix: Optional[str] = None

class ScheduleCreateRequest(BaseModel):
    cron_expr: str
    config_type: str  # "streaming" or "backfill"
    job_config: dict  # The full job configuration to publish

# =============================
# Helper Functions
# =============================
def publish_job_from_config(job_config: dict) -> dict:
    """
    Helper function to publish a job to RabbitMQ from a config dict.
    Used by scheduler to trigger jobs.
    """
    try:
        # Convert dict to FetchJobRequest
        job = FetchJobRequest(**job_config)

        # Call the main publish logic (extract core logic)
        corrid = generate_corrid()
        job_message = build_job_message(job, corrid)

        # Publish to RabbitMQ
        params = pika.URLParameters(RABBIT_URL)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.queue_declare(queue=EXTRACT_QUEUE, durable=True)

        channel.basic_publish(
            exchange='',
            routing_key=EXTRACT_QUEUE,
            body=json.dumps(job_message),
            properties=pika.BasicProperties(delivery_mode=2)
        )

        connection.close()
        jobs_published.labels(job_type=job.mode).inc()  # Track job publication
        logger.info(f"Scheduler published {job.mode} job with corrid={corrid}")

        return {"status": "success", "corrid": corrid}
    except Exception as e:
        logger.error(f"Failed to publish scheduled job: {e}")
        return {"status": "error", "message": str(e)}

def build_job_message(job: FetchJobRequest, corrid: str) -> dict:
    """Build the RabbitMQ job message from a FetchJobRequest."""
    job_message = {
        "mode": job.mode,
        "corr_id": corrid,
        "source": "crash",
        "join_key": "crash_record_id"
    }

    # Build primary crashes config
    crashes_select = ",".join(job.crashes_columns) if job.crashes_columns else "*"
    primary = {
        "id": CRASHES_ID,
        "alias": "crashes",
        "select": crashes_select,
        "order": "crash_date, crash_record_id",
        "page_size": 2000
    }

    # Add where clause based on mode
    if job.mode == "streaming":
        since_days = job.since_days or 30
        primary["where_by"] = {"since_days": since_days}
    elif job.mode == "backfill":
        if not job.start_date or not job.end_date:
            raise ValueError("Backfill requires start_date and end_date")
        job_message["date_range"] = {
            "field": "crash_date",
            "start": f"{job.start_date}T{job.start_time}",
            "end": f"{job.end_date}T{job.end_time}"
        }

    job_message["primary"] = primary

    # Build enrichment datasets
    enrich = []
    if job.include_vehicles and job.vehicles_columns:
        vehicles_select = ",".join(job.vehicles_columns)
        enrich.append({
            "id": VEHICLES_ID,
            "alias": "vehicles",
            "select": vehicles_select
        })

    if job.include_people and job.people_columns:
        people_select = ",".join(job.people_columns)
        enrich.append({
            "id": PEOPLE_ID,
            "alias": "people",
            "select": people_select
        })

    job_message["enrich"] = enrich

    # Add batching and storage config
    job_message["batching"] = {
        "id_batch_size": 200,
        "max_workers": {"vehicles": 4, "people": 4}
    }
    job_message["storage"] = {
        "bucket": "raw-data",
        "prefix": "crash",
        "compress": True
    }

    return job_message

def get_minio_client():
    """Create MinIO client."""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_USER,
        secret_key=MINIO_PASS,
        secure=MINIO_SSL
    )

def check_service_health(host: str, port: int, service_name: str) -> dict:
    """Check if a service is reachable."""
    import socket
    try:
        sock = socket.create_connection((host, port), timeout=2)
        sock.close()
        return {"service": service_name, "status": "healthy", "message": "Running"}
    except Exception as e:
        return {"service": service_name, "status": "unhealthy", "message": str(e)}

def check_rabbitmq_health() -> dict:
    """Check RabbitMQ health."""
    try:
        params = pika.URLParameters(RABBIT_URL)
        connection = pika.BlockingConnection(params)
        connection.close()
        return {"service": "RabbitMQ", "status": "healthy", "message": "Running"}
    except Exception as e:
        return {"service": "RabbitMQ", "status": "unhealthy", "message": str(e)}

def check_minio_health() -> dict:
    """Check MinIO health."""
    try:
        client = get_minio_client()
        client.list_buckets()
        return {"service": "MinIO", "status": "healthy", "message": "Running"}
    except Exception as e:
        return {"service": "MinIO", "status": "unhealthy", "message": str(e)}

def check_rabbitmq_consumer(queue_name: str, service_name: str) -> dict:
    """Check if a service is consuming from RabbitMQ queue."""
    try:
        params = pika.URLParameters(RABBIT_URL)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()

        # Declare queue to ensure it exists (passive=True would fail if queue doesn't exist)
        try:
            queue = channel.queue_declare(queue=queue_name, durable=True, passive=True)
            consumer_count = queue.method.consumer_count
            message_count = queue.method.message_count

            connection.close()

            if consumer_count > 0:
                return {
                    "service": service_name,
                    "status": "healthy",
                    "message": f"Active ({consumer_count} consumer(s), {message_count} msg(s) queued)"
                }
            else:
                return {
                    "service": service_name,
                    "status": "idle",
                    "message": f"No active consumers ({message_count} msg(s) queued)"
                }
        except Exception as queue_err:
            connection.close()
            return {
                "service": service_name,
                "status": "unknown",
                "message": f"Queue not found: {queue_name}"
            }

    except Exception as e:
        return {
            "service": service_name,
            "status": "unhealthy",
            "message": f"Cannot check queue: {str(e)[:50]}"
        }

def generate_corrid() -> str:
    """Generate correlation ID with timestamp prefix."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_uuid}"

# =============================
# API Endpoints
# =============================

# ------- Health & Status -------
@app.get("/api/health")
async def get_health():
    """Check health of all services."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "services": [
            check_minio_health(),
            check_rabbitmq_health(),
            check_rabbitmq_consumer("extract", "Extractor"),
            check_rabbitmq_consumer("transform", "Transformer"),
            check_rabbitmq_consumer("clean", "Cleaner"),
        ]
    }

# ------- Schema & Columns -------
@app.get("/api/schema/columns")
async def get_schema_columns(dataset: str = Query(..., description="Dataset: crashes, vehicles, or people")):
    """Get available columns for a dataset from Socrata API."""
    dataset_ids = {
        "crashes": CRASHES_ID,
        "vehicles": VEHICLES_ID,
        "people": PEOPLE_ID
    }

    if dataset not in dataset_ids:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}")

    dataset_id = dataset_ids[dataset]
    url = f"{SOCRATA_BASE}/api/views/{dataset_id}/columns.json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        columns_data = response.json()

        # Extract column names and types
        columns = []
        for col in columns_data:
            columns.append({
                "name": col.get("fieldName", col.get("name", "")),
                "type": col.get("dataTypeName", "unknown"),
                "description": col.get("description", "")
            })

        return {"dataset": dataset, "columns": columns}
    except Exception as e:
        logger.error(f"Failed to fetch schema for {dataset}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch schema: {str(e)}")

# ------- Data Fetcher -------
@app.post("/api/fetch/publish")
async def publish_fetch_job(job: FetchJobRequest):
    """Publish a data fetch job to RabbitMQ."""
    try:
        corrid = generate_corrid()
        job_message = build_job_message(job, corrid)

        # Publish to RabbitMQ
        params = pika.URLParameters(RABBIT_URL)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.queue_declare(queue=EXTRACT_QUEUE, durable=True)

        channel.basic_publish(
            exchange='',
            routing_key=EXTRACT_QUEUE,
            body=json.dumps(job_message),
            properties=pika.BasicProperties(delivery_mode=2)  # persistent
        )

        connection.close()

        logger.info(f"Published {job.mode} job with corrid={corrid}")

        return {
            "status": "success",
            "corrid": corrid,
            "mode": job.mode,
            "message": f"Job queued successfully to {EXTRACT_QUEUE}",
            "job_config": job_message
        }

    except Exception as e:
        logger.error(f"Failed to publish job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to publish job: {str(e)}")

# ------- MinIO Management -------
@app.get("/api/minio/browse")
async def browse_minio_folders(bucket: str):
    """Browse top-level folders in a MinIO bucket."""
    try:
        client = get_minio_client()

        # Check if bucket exists
        if not client.bucket_exists(bucket):
            raise HTTPException(status_code=404, detail=f"Bucket '{bucket}' not found")

        # List all objects in bucket
        objects = client.list_objects(bucket, recursive=True)

        # Group by top-level prefix (folder)
        folders = {}
        for obj in objects:
            key = obj.object_name
            # Extract top-level folder (e.g., "crash/corr=20231010_123456/")
            parts = key.split("/")
            if len(parts) >= 2:
                # Assume structure is: prefix/corr=timestamp/files
                # We want to group by "prefix/corr=timestamp/"
                if len(parts) >= 3 and parts[1].startswith("corr="):
                    folder = f"{parts[0]}/{parts[1]}/"
                else:
                    # Fallback: just use first two levels
                    folder = "/".join(parts[:2]) + "/"

                if folder not in folders:
                    folders[folder] = {
                        "prefix": folder,
                        "count": 0,
                        "size": 0,
                        "last_modified": obj.last_modified
                    }

                folders[folder]["count"] += 1
                folders[folder]["size"] += obj.size

                # Track latest modification time
                if obj.last_modified and obj.last_modified > folders[folder]["last_modified"]:
                    folders[folder]["last_modified"] = obj.last_modified

        # Convert to list and sort by last_modified (newest first)
        folder_list = []
        for folder_data in folders.values():
            folder_list.append({
                "prefix": folder_data["prefix"],
                "count": folder_data["count"],
                "size_bytes": folder_data["size"],
                "size_mb": round(folder_data["size"] / (1024 * 1024), 2),
                "last_modified": folder_data["last_modified"].isoformat() if folder_data["last_modified"] else None
            })

        folder_list.sort(key=lambda x: x["last_modified"] or "", reverse=True)

        return {
            "bucket": bucket,
            "total_folders": len(folder_list),
            "folders": folder_list
        }

    except S3Error as e:
        raise HTTPException(status_code=500, detail=f"MinIO error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/minio/preview")
async def preview_minio_objects(bucket: str, prefix: str = ""):
    """Preview objects in MinIO bucket (dry-run for delete)."""
    try:
        client = get_minio_client()

        # Check if bucket exists
        if not client.bucket_exists(bucket):
            raise HTTPException(status_code=404, detail=f"Bucket '{bucket}' not found")

        # List objects
        objects = client.list_objects(bucket, prefix=prefix, recursive=True)
        object_list = []
        total_size = 0

        for obj in objects:
            object_list.append({
                "key": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None
            })
            total_size += obj.size

        # Group by folder (corrid)
        folders = {}
        for obj in object_list:
            key = obj["key"]
            # Extract folder from key (e.g., "crash/corr=20231010_123456/")
            parts = key.split("/")
            if len(parts) > 1:
                folder = "/".join(parts[:-1]) + "/"
                if folder not in folders:
                    folders[folder] = {"count": 0, "size": 0, "keys": []}
                folders[folder]["count"] += 1
                folders[folder]["size"] += obj["size"]
                folders[folder]["keys"].append(key)

        return {
            "bucket": bucket,
            "prefix": prefix,
            "total_objects": len(object_list),
            "total_size_bytes": total_size,
            "folders": [{"folder": k, **v} for k, v in folders.items()],
            "objects": object_list[:100]  # Limit to first 100 for preview
        }

    except S3Error as e:
        raise HTTPException(status_code=500, detail=f"MinIO error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.delete("/api/minio/folder")
async def delete_minio_folder(request: MinioDeleteRequest):
    """Delete objects in MinIO by prefix (folder)."""
    logger.info(f"DELETE folder request received: bucket={request.bucket}, prefix={request.prefix}")
    try:
        client = get_minio_client()

        if not client.bucket_exists(request.bucket):
            logger.error(f"Bucket not found: {request.bucket}")
            raise HTTPException(status_code=404, detail=f"Bucket '{request.bucket}' not found")

        if not request.prefix:
            logger.error("No prefix provided for folder deletion")
            raise HTTPException(status_code=400, detail="Prefix is required for folder deletion")

        # List and delete objects
        logger.info(f"Listing objects in {request.bucket}/{request.prefix}")
        objects_to_delete = client.list_objects(request.bucket, prefix=request.prefix, recursive=True)
        deleted_count = 0

        for obj in objects_to_delete:
            logger.debug(f"Deleting: {obj.object_name}")
            client.remove_object(request.bucket, obj.object_name)
            deleted_count += 1

        logger.info(f"Successfully deleted {deleted_count} objects from {request.bucket}/{request.prefix}")

        return {
            "status": "success",
            "bucket": request.bucket,
            "prefix": request.prefix,
            "deleted_count": deleted_count
        }

    except Exception as e:
        logger.error(f"Failed to delete folder: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")

@app.delete("/api/minio/bucket")
async def delete_minio_bucket(request: MinioDeleteRequest):
    """Wipe all contents from MinIO bucket (but keep the bucket)."""
    logger.info(f"WIPE bucket request received: bucket={request.bucket}")
    try:
        client = get_minio_client()

        if not client.bucket_exists(request.bucket):
            logger.error(f"Bucket not found: {request.bucket}")
            raise HTTPException(status_code=404, detail=f"Bucket '{request.bucket}' not found")

        # Delete all objects in the bucket
        logger.info(f"Listing all objects in bucket: {request.bucket}")
        objects = client.list_objects(request.bucket, recursive=True)
        deleted_count = 0
        for obj in objects:
            logger.debug(f"Deleting: {obj.object_name}")
            client.remove_object(request.bucket, obj.object_name)
            deleted_count += 1

        # Do NOT delete the bucket itself - just wipe its contents
        logger.info(f"Successfully wiped {deleted_count} objects from bucket '{request.bucket}'")

        return {
            "status": "success",
            "bucket": request.bucket,
            "deleted_objects": deleted_count,
            "message": f"Bucket '{request.bucket}' wiped successfully ({deleted_count} objects removed)"
        }

    except Exception as e:
        logger.error(f"Failed to wipe bucket: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to wipe bucket: {str(e)}")

# ------- Gold/DuckDB Management -------
@app.get("/api/gold/status")
async def get_gold_status():
    """Get DuckDB Gold database status."""
    try:
        if not Path(GOLD_DB_PATH).exists():
            return {
                "exists": False,
                "path": GOLD_DB_PATH,
                "message": "Gold database does not exist"
            }

        file_size = Path(GOLD_DB_PATH).stat().st_size

        con = duckdb.connect(GOLD_DB_PATH, read_only=True)

        # Get table list
        tables_result = con.execute("SHOW TABLES").fetchall()
        tables = [row[0] for row in tables_result]

        # Get row counts per table
        table_stats = []
        total_rows = 0
        for table in tables:
            count_result = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            count = count_result[0] if count_result else 0
            total_rows += count
            table_stats.append({"table": table, "row_count": count})

        con.close()

        return {
            "exists": True,
            "path": GOLD_DB_PATH,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "total_tables": len(tables),
            "total_rows": total_rows,
            "tables": table_stats
        }

    except Exception as e:
        logger.error(f"Failed to get gold status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.get("/api/gold/peek")
async def peek_gold_table(
    table: str = Query('"gold"."main"."crashes"', description="Table name"),
    columns: Optional[str] = Query(None, description="Comma-separated column names"),
    limit: int = Query(50, ge=1, le=200, description="Row limit")
):
    """Quick peek at Gold table data."""
    try:
        if not Path(GOLD_DB_PATH).exists():
            raise HTTPException(status_code=404, detail="Gold database does not exist")

        con = duckdb.connect(GOLD_DB_PATH, read_only=True)

        # Get column names if not specified
        if columns:
            select_cols = columns
        else:
            # Get first 8 columns
            schema_result = con.execute(f"DESCRIBE {table}").fetchall()
            col_names = [row[0] for row in schema_result[:8]]
            select_cols = ",".join(col_names)

        # Query data
        query = f"SELECT {select_cols} FROM {table} LIMIT {limit}"
        result = con.execute(query).fetchdf()

        con.close()

        # Replace NaN and Inf values to make JSON serializable
        import numpy as np
        result = result.replace([np.inf, -np.inf], np.nan)

        # Convert to dict and replace NaN with None
        data = result.to_dict(orient="records")
        # Replace NaN values with None for JSON compatibility
        for row in data:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None

        # Convert to JSON-serializable format
        return {
            "table": table,
            "columns": select_cols.split(","),
            "row_count": len(result),
            "data": data
        }

    except Exception as e:
        logger.error(f"Failed to peek table: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to peek table: {str(e)}")

@app.delete("/api/gold/wipe")
async def wipe_gold_database():
    """Delete Gold database file completely."""
    try:
        if Path(GOLD_DB_PATH).exists():
            Path(GOLD_DB_PATH).unlink()
            logger.info(f"Wiped Gold database: {GOLD_DB_PATH}")
            return {
                "status": "success",
                "message": f"Gold database deleted: {GOLD_DB_PATH}"
            }
        else:
            return {
                "status": "success",
                "message": "Gold database does not exist (already clean)"
            }
    except Exception as e:
        logger.error(f"Failed to wipe gold database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to wipe: {str(e)}")

@app.get("/api/gold/query")
async def query_gold(sql: str = Query(..., description="SQL query to execute")):
    """Execute custom SQL query on Gold database."""
    try:
        if not Path(GOLD_DB_PATH).exists():
            raise HTTPException(status_code=404, detail="Gold database does not exist")

        con = duckdb.connect(GOLD_DB_PATH, read_only=True)
        result = con.execute(sql).fetchdf()
        con.close()

        # Replace NaN and Inf values to make JSON serializable
        import numpy as np
        import json

        # Replace inf/-inf with NaN, then NaN with None
        result = result.replace([np.inf, -np.inf], np.nan)

        # Convert to JSON string with NaN handling, then back to dict
        json_str = result.to_json(orient="records", force_ascii=False, date_format='iso')
        data = json.loads(json_str)

        return {
            "query": sql,
            "row_count": len(result),
            "data": data
        }

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=400, detail=f"Query failed: {str(e)}")

# ------- Reports -------
def get_run_details(client, corrid: str) -> dict:
    """Get detailed information about a specific run."""
    details = {
        "corrid": corrid,
        "rows_processed": 0,
        "status": "unknown",
        "mode": "unknown"
    }

    # Check if run made it to transform stage (has processed data)
    try:
        transform_objects = list(client.list_objects("transform-data", prefix=f"crash/corr={corrid}/", recursive=True))
        if transform_objects:
            details["status"] = "success"
            # Try to count rows from merged.csv if it exists
            for obj in transform_objects:
                if obj.object_name.endswith("merged.csv"):
                    try:
                        # Get object and count lines (rough approximation)
                        response = client.get_object("transform-data", obj.object_name)
                        content = response.read().decode('utf-8')
                        lines = content.strip().split('\n')
                        details["rows_processed"] = len(lines) - 1  # Exclude header
                        response.close()
                        response.release_conn()
                    except:
                        pass
        else:
            details["status"] = "no data"
    except:
        details["status"] = "failed"

    return details

@app.get("/api/reports/summary")
async def get_reports_summary():
    """Get summary statistics for reports tab."""
    try:
        # Extract correlation IDs from MinIO folder structure
        corrids = set()
        latest_corrid = None
        latest_timestamp = None

        try:
            client = get_minio_client()

            # Check _runs/ directory in raw-data bucket for all pipeline runs
            if client.bucket_exists("raw-data"):
                objects = client.list_objects("raw-data", prefix="_runs/corr=", recursive=False)
                for obj in objects:
                    # Extract corrid from folder name: _runs/corr=YYYYMMDD_HHMMSS_uuid/
                    if "/corr=" in obj.object_name:
                        parts = obj.object_name.split("/corr=")
                        if len(parts) > 1:
                            corrid = parts[1].rstrip("/").split("/")[0]
                            corrids.add(corrid)

                            # Extract timestamp from corrid (format: YYYYMMDD_HHMMSS_uuid)
                            try:
                                timestamp_str = "_".join(corrid.split("_")[:2])  # Get YYYYMMDD_HHMMSS
                                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                                if latest_timestamp is None or timestamp > latest_timestamp:
                                    latest_timestamp = timestamp
                                    latest_corrid = corrid
                            except:
                                pass
        except Exception as e:
            logger.warning(f"Could not extract corrids from MinIO: {e}")

        # Get Gold database stats
        gold_row_count = 0
        latest_data_date = None

        if Path(GOLD_DB_PATH).exists():
            con = duckdb.connect(GOLD_DB_PATH, read_only=True)

            # Get total rows in main table
            try:
                gold_row_count = con.execute('SELECT COUNT(*) FROM "gold"."main"."crashes"').fetchone()[0]
            except:
                pass

            # Get latest crash date
            try:
                latest_data_date = con.execute('SELECT MAX(crash_date) FROM "gold"."main"."crashes"').fetchone()[0]
            except:
                pass

            con.close()

        return {
            "total_runs": len(corrids) if corrids else 0,
            "latest_corrid": latest_corrid if latest_corrid else "None",
            "gold_row_count": gold_row_count,
            "latest_data_date": str(latest_data_date) if latest_data_date else None,
            "last_run_timestamp": latest_timestamp.isoformat() if latest_timestamp else None
        }

    except Exception as e:
        logger.error(f"Failed to get summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@app.get("/api/reports/run-history")
async def get_run_history():
    """Get detailed run history for all pipeline executions."""
    try:
        client = get_minio_client()
        run_history = []

        # Get all correlation IDs from _runs/ directory
        if client.bucket_exists("raw-data"):
            objects = client.list_objects("raw-data", prefix="_runs/corr=", recursive=False)
            for obj in objects:
                if "/corr=" in obj.object_name:
                    parts = obj.object_name.split("/corr=")
                    if len(parts) > 1:
                        corrid = parts[1].rstrip("/").split("/")[0]

                        # Extract timestamp
                        try:
                            timestamp_str = "_".join(corrid.split("_")[:2])
                            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                            timestamp_formatted = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            timestamp_formatted = "Unknown"
                            timestamp = None

                        # Get detailed info about the run
                        details = get_run_details(client, corrid)

                        run_history.append({
                            "corrid": corrid,
                            "timestamp": timestamp_formatted,
                            "rows_processed": details["rows_processed"],
                            "status": details["status"],
                            "mode": details["mode"]
                        })

            # Sort by timestamp (newest first)
            run_history.sort(key=lambda x: x["timestamp"], reverse=True)

        return {
            "total_runs": len(run_history),
            "runs": run_history
        }

    except Exception as e:
        logger.error(f"Failed to get run history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get run history: {str(e)}")

# ------- Scheduler -------
@app.post("/api/schedule/create")
async def create_schedule(request: ScheduleCreateRequest):
    """Create a new scheduled job."""
    try:
        schedule = sched_module.add_schedule(
            cron_expr=request.cron_expr,
            config_type=request.config_type,
            job_config=request.job_config
        )

        return {
            "status": "success",
            "message": "Schedule created successfully",
            "schedule": schedule.to_dict()
        }
    except Exception as e:
        logger.error(f"Failed to create schedule: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create schedule: {str(e)}")

@app.get("/api/schedule/list")
async def list_schedules():
    """List all schedules."""
    try:
        schedules = sched_module.get_schedules()
        return {
            "schedules": [s.to_dict() for s in schedules],
            "total": len(schedules)
        }
    except Exception as e:
        logger.error(f"Failed to list schedules: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list schedules: {str(e)}")

@app.delete("/api/schedule/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """Delete a schedule."""
    try:
        success = sched_module.remove_schedule(schedule_id)
        if success:
            return {
                "status": "success",
                "message": f"Schedule {schedule_id} deleted"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Schedule {schedule_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete schedule: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete schedule: {str(e)}")

@app.put("/api/schedule/{schedule_id}/toggle")
async def toggle_schedule(schedule_id: str, enabled: bool = Query(..., description="Enable or disable")):
    """Enable or disable a schedule."""
    try:
        schedule = sched_module.toggle_schedule(schedule_id, enabled)
        if schedule:
            return {
                "status": "success",
                "message": f"Schedule {schedule_id} {'enabled' if enabled else 'disabled'}",
                "schedule": schedule.to_dict()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Schedule {schedule_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle schedule: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to toggle schedule: {str(e)}")

# ------- Model Endpoints -------
def load_model():
    """Load the ML model from disk and cache it."""
    global _model_cache
    if _model_cache is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

class PredictFromGoldRequest(BaseModel):
    start_date: str
    end_date: str
    max_rows: int = 5000

@app.get("/api/model/info")
async def get_model_info():
    """Get model metadata and information."""
    try:
        model = load_model()

        # Extract model info
        model_type = type(model).__name__

        # Get the base estimator (pipeline inside CalibratedClassifierCV)
        base_estimator_type = "Unknown"
        if hasattr(model, 'estimator'):
            base_estimator_type = type(model.estimator).__name__

        # Expected features
        numerical_features = [
            'posted_speed_limit', 'num_units', 'crash_hour', 'crash_day_of_week',
            'crash_month', 'beat_of_occurrence', 'intersection_related_i', 'hit_and_run_i',
            'latitude', 'longitude', 'lat_bin', 'lng_bin', 'veh_count', 'ppl_count',
            'year', 'month', 'day', 'day_of_week', 'hour', 'is_weekend',
            'veh_truck_i', 'veh_mc_i', 'ppl_age_mean', 'ppl_age_min', 'ppl_age_max'
        ]

        categorical_features = [
            'traffic_control_device', 'device_condition', 'weather_condition',
            'lighting_condition', 'first_crash_type', 'trafficway_type',
            'alignment', 'roadway_surface_cond', 'road_defect', 'hour_bin'
        ]

        return {
            "model_type": model_type,
            "base_estimator": base_estimator_type,
            "decision_threshold": 0.3449,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "total_features": len(numerical_features) + len(categorical_features),
            "target": "injury",
            "description": "CalibratedClassifierCV wrapping Pipeline with XGBoost. Preprocessing (imputation, scaling, one-hot encoding) is handled inside the pipeline."
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/api/model/predict")
async def predict(request: PredictRequest):
    """Run predictions on input data."""
    try:
        prediction_start = time.time()
        model = load_model()

        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)

        # Replace NaN and Inf values before prediction
        import numpy as np
        df = df.replace([np.inf, -np.inf], np.nan)

        # Get predictions and probabilities
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)

        # Track prediction metrics
        model_predictions.inc(len(predictions))
        prediction_latency.observe(time.time() - prediction_start)

        # Apply decision threshold (0.3449)
        threshold = 0.3449
        proba_positive = probabilities[:, 1]  # Probability of class 1 (injury)
        predictions_thresholded = (proba_positive >= threshold).astype(int)

        # Convert to lists for JSON serialization, handling NaN values
        predictions_list = [int(p) if not np.isnan(p) else None for p in predictions]
        predictions_thresholded_list = [int(p) if not np.isnan(p) else None for p in predictions_thresholded]
        probabilities_list = [[float(p[0]) if not np.isnan(p[0]) else None,
                               float(p[1]) if not np.isnan(p[1]) else None] for p in probabilities]

        return {
            "predictions": predictions_list,
            "predictions_thresholded": predictions_thresholded_list,
            "probabilities": probabilities_list,
            "threshold": threshold,
            "count": len(predictions)
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/api/model/predict-from-gold")
async def predict_from_gold(request: PredictFromGoldRequest):
    """Query Gold DB, run predictions, and return results directly (no data transfer)."""
    try:
        import numpy as np

        # Check if Gold DB exists
        if not Path(GOLD_DB_PATH).exists():
            raise HTTPException(status_code=404, detail="Gold database does not exist")

        # Load model
        model = load_model()

        # Define required features
        numerical_features = [
            'posted_speed_limit', 'num_units', 'crash_hour', 'crash_day_of_week',
            'crash_month', 'beat_of_occurrence', 'intersection_related_i', 'hit_and_run_i',
            'latitude', 'longitude', 'lat_bin', 'lng_bin', 'veh_count', 'ppl_count',
            'year', 'month', 'day', 'day_of_week', 'hour', 'is_weekend',
            'veh_truck_i', 'veh_mc_i', 'ppl_age_mean', 'ppl_age_min', 'ppl_age_max'
        ]
        categorical_features = [
            'traffic_control_device', 'device_condition', 'weather_condition',
            'lighting_condition', 'first_crash_type', 'trafficway_type',
            'alignment', 'roadway_surface_cond', 'road_defect', 'hour_bin'
        ]

        all_features = numerical_features + categorical_features
        feature_str = ", ".join([f'"{f}"' for f in all_features])

        # Query Gold DB
        con = duckdb.connect(GOLD_DB_PATH, read_only=True)
        query = f"""
        SELECT {feature_str}, "injury"
        FROM "gold"."main"."crashes"
        WHERE crash_date BETWEEN '{request.start_date}' AND '{request.end_date}'
        LIMIT {request.max_rows}
        """

        df = con.execute(query).fetchdf()
        con.close()

        if df.empty:
            return {
                "success": False,
                "message": "No data found for the specified date range",
                "row_count": 0
            }

        # Check if injury column exists for metrics
        has_labels = 'injury' in df.columns
        y_true = df['injury'].values if has_labels else None

        # Prepare features for prediction
        X = df[all_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        # Get predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        # Apply decision threshold
        threshold = 0.3449
        proba_positive = probabilities[:, 1]
        predictions_thresholded = (proba_positive >= threshold).astype(int)

        # Calculate metrics if labels exist
        metrics = None
        if has_labels and y_true is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

            # Remove rows where y_true is NaN
            valid_idx = ~np.isnan(y_true)
            if valid_idx.sum() > 0:
                y_true_valid = y_true[valid_idx].astype(int)
                y_pred_valid = predictions_thresholded[valid_idx]

                metrics = {
                    "accuracy": float(accuracy_score(y_true_valid, y_pred_valid)),
                    "precision": float(precision_score(y_true_valid, y_pred_valid, zero_division=0)),
                    "recall": float(recall_score(y_true_valid, y_pred_valid, zero_division=0)),
                    "f1_score": float(f1_score(y_true_valid, y_pred_valid, zero_division=0)),
                    "confusion_matrix": confusion_matrix(y_true_valid, y_pred_valid).tolist()
                }

        # Prediction distribution
        unique, counts = np.unique(predictions_thresholded, return_counts=True)
        pred_distribution = {0: 0, 1: 0}  # Initialize both classes
        for k, v in zip(unique, counts):
            pred_distribution[int(k)] = int(v)

        logger.info(f"Prediction distribution: {pred_distribution}")

        # Probability statistics
        prob_stats = {
            "min": float(np.min(proba_positive)),
            "max": float(np.max(proba_positive)),
            "mean": float(np.mean(proba_positive)),
            "median": float(np.median(proba_positive)),
            "std": float(np.std(proba_positive))
        }

        # Prepare probability histogram data (binned)
        prob_hist, bin_edges = np.histogram(proba_positive, bins=50, range=(0, 1))
        probability_histogram = {
            "counts": prob_hist.tolist(),
            "bin_edges": bin_edges.tolist()
        }

        return {
            "success": True,
            "row_count": len(df),
            "predictions": predictions_thresholded.tolist(),
            "probabilities": proba_positive.tolist(),
            "prediction_distribution": pred_distribution,
            "probability_stats": prob_stats,
            "probability_histogram": probability_histogram,
            "threshold": threshold,
            "metrics": metrics,
            "has_labels": has_labels
        }

    except Exception as e:
        logger.error(f"Prediction from Gold failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ------- PDF Reports -------
@app.get("/api/reports/download/pdf")
async def download_pdf_report():
    """Generate and download comprehensive PDF report."""
    try:
        # Get summary data
        summary_data = {}
        corrids = set()
        latest_corrid = None
        latest_timestamp = None

        # Extract correlation IDs from MinIO (_runs/ directory)
        try:
            client = get_minio_client()
            if client.bucket_exists("raw-data"):
                objects = client.list_objects("raw-data", prefix="_runs/corr=", recursive=False)
                for obj in objects:
                    if "/corr=" in obj.object_name:
                        parts = obj.object_name.split("/corr=")
                        if len(parts) > 1:
                            corrid = parts[1].rstrip("/").split("/")[0]
                            corrids.add(corrid)
                            try:
                                timestamp_str = "_".join(corrid.split("_")[:2])
                                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                                if latest_timestamp is None or timestamp > latest_timestamp:
                                    latest_timestamp = timestamp
                                    latest_corrid = corrid
                            except:
                                pass
        except Exception as e:
            logger.warning(f"Could not extract corrids from MinIO: {e}")

        # Get Gold database stats
        gold_row_count = 0
        latest_data_date = None

        if Path(GOLD_DB_PATH).exists():
            con = duckdb.connect(GOLD_DB_PATH, read_only=True)
            try:
                gold_row_count = con.execute('SELECT COUNT(*) FROM "gold"."main"."crashes"').fetchone()[0]
            except:
                pass
            try:
                latest_data_date = con.execute('SELECT MAX(crash_date) FROM "gold"."main"."crashes"').fetchone()[0]
            except:
                pass
            con.close()

        summary_data = {
            "total_runs": len(corrids) if corrids else 0,
            "latest_corrid": latest_corrid if latest_corrid else "None",
            "gold_row_count": gold_row_count,
            "latest_data_date": str(latest_data_date) if latest_data_date else None,
            "last_run_timestamp": latest_timestamp.isoformat() if latest_timestamp else None
        }

        # Get detailed run history
        run_history = []
        for corrid in list(corrids)[:10]:  # Last 10 runs
            details = get_run_details(client, corrid)
            # Extract timestamp
            try:
                timestamp_str = "_".join(corrid.split("_")[:2])
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                timestamp_formatted = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except:
                timestamp_formatted = "Unknown"

            run_history.append({
                "corrid": corrid,
                "timestamp": timestamp_formatted,
                "mode": details["mode"],
                "rows": details["rows_processed"],
                "status": details["status"]
            })

        # Generate PDF
        pdf_buffer = report_generator.generate_run_history_pdf(summary_data, run_history)

        # Return as streaming response
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=crash_etl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )

    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

# ------- Root -------
@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Chicago Crash ETL API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "schema": "/api/schema/columns?dataset=crashes",
            "fetch": "/api/fetch/publish",
            "minio": "/api/minio/*",
            "gold": "/api/gold/*",
            "model": "/api/model/*",
            "reports": "/api/reports/*"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
