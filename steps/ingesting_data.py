from src import logger
import pandas as pd
from zenml import step
from typing import Annotated

class IngestData:
    """A class to handle data ingestion from a CSV file."""
    def __init__(self, data_path: str):
        """
        Args :
            data_path (str): The path to the CSV file to be ingested.
        """
        self.data_path = data_path

    def get_data(self):
        """
        Reads data from the specified CSV file and returns it as a pandas DataFrame.
        """
        logger.info(f"Reading data from {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_data(data_path: str) -> Annotated[pd.DataFrame, "Ingested_data"]:
    """
    Ingests data from a CSV file and returns it as a pandas DataFrame.

    Args:
        data_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The ingested data as a pandas DataFrame.
    """
    try:
        ingest_data_instance = IngestData(data_path)
        df = ingest_data_instance.get_data()
        return df
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        raise