import argparse
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd
import requests
from pandas import DataFrame
from pydantic import BaseModel
from ragas import Dataset
from requests import Response


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
    dataset_samples = cast(list[dict[str, Any]], dataframe.to_dict(orient="records"))

    # Create Ragas Dataset
    dataset: Dataset[BaseModel] = Dataset(
        name="ragas_dataset",
        data=dataset_samples,
        backend="local/jsonl",
        root_dir="./data",
    )

    # Save the dataset
    dataset.save()


def get_converter(url: str) -> Callable[[BytesIO], DataFrame]:
    """Extract the file format from the URL and return the converter function"""
    suffix = Path(url).suffix.lower()

    format_map: dict[str, Callable[[BytesIO], DataFrame]] = {
        ".json": pd.read_json,
        ".csv": custom_convert_csv,
        ".parquet": pd.read_parquet,
        ".prq": pd.read_parquet,
    }

    if suffix in format_map:
        return format_map[suffix]

    raise TypeError(f"Unsupported filetype at url: {url}")


def custom_convert_csv(input_file: BytesIO) -> DataFrame:
    """
    Converts a CSV input file to a Pandas DataFrame and, if it exists, turns 'retrieved_contexts' into a list
    (RAGAS requires 'retrieved_contexts' as a list of strings)

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


def main(url: str) -> None:
    """Download provided dataset -> convert to Ragas dataset -> save to data/ragas_dataset.jsonl

    Source dataset must contain columns: user_input, retrieved_contexts, reference
    """
    converter = get_converter(url)

    # Download file from URL and raise HTTP error if it occurs
    file: Response = requests.get(url, timeout=20)
    file.raise_for_status()

    # Load into DataFrame by using the correct converter
    buffer = BytesIO(file.content)
    dataframe = converter(buffer)

    # Convert DataFrame to Ragas dataset and save it
    dataframe_to_ragas_dataset(dataframe)


if __name__ == "__main__":
    # Parse parameter the script was called with (URL)
    parser = argparse.ArgumentParser(
        description="Download provided dataset -> convert to Ragas dataset -> save to data/datasets/ragas_dataset.jsonl"
    )
    parser.add_argument(
        "url",
        help="URL to the dataset in .csv / .json / .parquet format (must have user_input, retrieved_contexts, and reference columns)",
    )
    args = parser.parse_args()

    # Call main using the parsed URL
    main(args.url)
