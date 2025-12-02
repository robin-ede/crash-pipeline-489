"""
MinIO I/O utilities for reading Silver CSV files.
"""
import io
import logging
from minio import Minio
import polars as pl

logging.basicConfig(level=logging.INFO, format="[cleaner:minio_io] %(message)s")


def minio_client(endpoint: str, access_key: str, secret_key: str, secure: bool = False) -> Minio:
    """Create MinIO client."""
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def read_csv_from_minio(cli: Minio, bucket: str, key: str) -> pl.DataFrame:
    """
    Download a CSV file from MinIO and return as Polars DataFrame.

    Args:
        cli: MinIO client
        bucket: Bucket name
        key: Object key (path)

    Returns:
        Polars DataFrame

    Raises:
        Exception: If object not found or read fails
    """
    resp = None
    try:
        logging.info(f"Reading s3://{bucket}/{key}")
        resp = cli.get_object(bucket, key)
        data = resp.read()
        return pl.read_csv(io.BytesIO(data))
    finally:
        if resp is not None:
            try:
                resp.close()
                resp.release_conn()
            except Exception:
                pass
