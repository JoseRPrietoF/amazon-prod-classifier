"""Module for raw data extraction in an ETL pipeline."""

from pyspark.sql import SparkSession, DataFrame
import logging
import yaml


def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with INFO level.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)


logger = setup_logger(__name__)


def load_jsonl_gz(path: str, fraction: float = 1.0, seed: int = 42) -> DataFrame:
    """
    Load JSON objects from a gzip-compressed file into a Spark DataFrame.

    Args:
        path (str): Path to the JSONL GZIP file.
        fraction (float, optional): Fraction of data to sample (0.0 to 1.0). Defaults to 1.0.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        DataFrame: Spark DataFrame with loaded data.

    Raises:
        ValueError: If fraction is not in the range [0, 1].
        Exception: If the file loading fails.
    """
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("Fraction must be between 0.0 and 1.0")

    try:
        spark = create_spark_session()
        df = spark.read.json(path)

        if fraction < 1.0:
            df = df.sample(fraction=fraction, seed=seed)

        logger.info("Successfully loaded JSONL GZIP file: %s", path)
        return df
    except Exception as e:
        logger.error("Failed to load JSONL GZIP file: %s", e, exc_info=True)
        raise


def load_parquet(path: str) -> DataFrame:
    """
    Load a Parquet file into a Spark DataFrame.

    Args:
        path (str): Path to the Parquet file.

    Returns:
        DataFrame: Spark DataFrame with loaded data.

    Raises:
        Exception: If the file loading fails.
    """
    try:
        spark = create_spark_session()
        df = spark.read.parquet(path)
        logger.info("Successfully loaded Parquet file: %s", path)
        return df
    except Exception as e:
        logger.error("Failed to load Parquet file: %s", e, exc_info=True)
        raise


def create_spark_session() -> SparkSession:
    """
    Create or retrieve a Spark session with predefined configurations.

    Returns:
        SparkSession: Configured Spark session.

    Raises:
        Exception: If the Spark session creation fails.
    """
    try:
        spark = (
            SparkSession.builder.appName("ProductClassifier")
            .master("local[*]")
            .config("spark.driver.memory", "6g")
            .config("spark.executor.memory", "6g")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        return spark
    except Exception as e:
        logger.error("Failed to create Spark session: %s", e, exc_info=True)
        raise


def read_yaml_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error("Failed to load configuration file: %s", e, exc_info=True)
        raise
