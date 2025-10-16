import argparse
import requests
import json

from pathlib import Path
from typing import Callable
from io import BytesIO

import pandas as pd
from pandas import DataFrame

def dataframe_to_json(dataframe: DataFrame) -> None:
    """Convert DataFrame to JSON and save to data/dataset.json."""

    # Set output directory (and create it if it doesn't exist already)
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)

    # Wrap the data in a "dataset" key
    output_data = {
        "dataset": dataframe.to_dict(orient="records")
    }

    with open(output_dir / 'dataset.json', 'w') as file:
        json.dump(output_data, file, indent=2)


def get_converter(url: str) -> Callable[[BytesIO], DataFrame]:
    """Extract the file format from the URL and return the converter function"""
    suffix = Path(url).suffix.lower()

    format_map = {
        '.json': pd.read_json,
        '.csv': pd.read_csv,
        '.parquet': pd.read_parquet,
        '.prq': pd.read_parquet,
    }

    if suffix in format_map:
        return format_map[suffix]

    raise TypeError(f"Unsupported filetype at url: {url}")


def main(url : str) -> None:
    """Download provided dataset -> convert to json -> save to data/dataset.json"""
    converter = get_converter(url)

    # Download file from URL and raise HTTP error if it occurs
    file = requests.get(url)
    file.raise_for_status()

    # Load into DataFrame by using the correct converter
    buffer = BytesIO(file.content)
    dataframe = converter(buffer)

    # Convert DataFrame to JSON and save it in data/dataset.json
    dataframe_to_json(dataframe)

if __name__ == '__main__':
    # Parse parameter the script was called with (URL)
    parser = argparse.ArgumentParser(description="Download provided dataset -> convert to json -> save to data/dataset.json")
    parser.add_argument('url', help='URL to the dataset in .csv / .json / .parquet format')
    args = parser.parse_args()

    # Call main using the parsed URL
    main(args.url)
