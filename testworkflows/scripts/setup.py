import argparse
import logging
import os
from pathlib import Path
from typing import Callable
from io import BytesIO

import boto3
from botocore.client import Config
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel
from ragas import Dataset

# Set up module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def dataframe_to_ragas_dataset(dataframe: DataFrame) -> None:
    """Convert DataFrame to Ragas dataset and save to data/ragas_dataset.jsonl.

    Expected schema:
    - user_input: The test question/prompt
    - retrieved_contexts: List of retrieved context strings
    - reference: The reference/ground truth answer
    """

    # Set output directory (and create it if it doesn't exist already)
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Convert DataFrame to list of dictionaries
    dataset_samples = dataframe.to_dict(orient="records")

    # Create Ragas Dataset
    dataset: Dataset[BaseModel] = Dataset(
        name="ragas_dataset",
        data=dataset_samples,
        backend="local/jsonl",
        root_dir="./data",
    )

    # Save the dataset
    dataset.save()


def get_converter(key: str) -> Callable[[BytesIO], DataFrame]:
    """Extract the file format from the S3 key and return the converter function"""
    suffix = Path(key).suffix.lower()

    format_map: dict[str, Callable[[BytesIO], DataFrame]] = {
        ".json": pd.read_json,
        ".csv": custom_convert_csv,
        ".parquet": pd.read_parquet,
        ".prq": pd.read_parquet,
    }

    if suffix in format_map:
        return format_map[suffix]

    raise TypeError(f"Unsupported filetype for key: {key}")


def custom_convert_csv(input_file: BytesIO) -> DataFrame:
    """
    Converts a CSV input file to a Pandas DataFrame and, if it exists, turns 'retrieved_contexts' into a list (RAGAS requires 'retrieved_contexts' as a list of strings)

    Args:
        input_file: The CSV input_file

    Returns:
        Pandas DataFrame with correct formatting
    """

    dataframe: DataFrame = pd.read_csv(input_file)

    # Ensure retrieved_contexts is a list (convert string to list if needed)
    if "retrieved_contexts" in dataframe:
        dataframe["retrieved_contexts"] = dataframe["retrieved_contexts"].apply(
            lambda x: x if isinstance(x, list) else [x] if x else []
        )

    return dataframe


def create_s3_client() -> boto3.client:
    """Create and configure S3 client for MinIO"""
    # Get MinIO credentials from environment
    access_key = os.getenv("MINIO_ROOT_USER", "minio")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD", "minio123")
    endpoint_url = os.getenv("MINIO_ENDPOINT", "http://testkube-minio-testbench.testbench:9000")

    logger.info(f"Connecting to MinIO at {endpoint_url}")

    # Create S3 client with MinIO configuration
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",  # MinIO doesn't care about region, but boto3 requires it
    )

    return s3_client


def main(bucket: str, key: str) -> None:
    """Download dataset from S3 -> convert to Ragas dataset -> save to data/ragas_dataset.jsonl

    Source dataset must contain columns: user_input, retrieved_contexts, reference
    """
    logger.info(f"Starting dataset download from S3: s3://{bucket}/{key}")

    # Get converter based on file extension
    converter = get_converter(key)

    # Create S3 client
    s3_client = create_s3_client()

    # Download file from S3
    logger.info(f"Downloading from bucket '{bucket}', key '{key}'...")
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        file_content = response["Body"].read()
        logger.info(f"Downloaded {len(file_content)} bytes")
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        raise

    # Load into DataFrame by using the correct converter
    logger.info("Converting to DataFrame...")
    buffer = BytesIO(file_content)
    dataframe = converter(buffer)
    logger.info(f"Loaded {len(dataframe)} rows")

    # Convert DataFrame to Ragas dataset and save it
    logger.info("Converting to Ragas dataset...")
    dataframe_to_ragas_dataset(dataframe)
    logger.info("✓ Dataset saved successfully to data/ragas_dataset.jsonl")


if __name__ == "__main__":


    # Call main using the parsed bucket and key
    main("datasets", "dataset.json")
