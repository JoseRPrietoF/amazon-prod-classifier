import pytest
from pyspark.sql import SparkSession
from transformers import PreTrainedTokenizerBase
from src.data.etl import (
    etl_pipeline,
    read_yaml_config,
    tokenize_and_transform,
    load_tokenizer,
)

PATH_YAML = "src/tests/config_test.yaml"


@pytest.fixture(scope="module")
def spark():
    """
    PySpark session fixture for testing.
    """
    return SparkSession.builder.appName("ETLTest").master("local[*]").getOrCreate()


@pytest.fixture(scope="module")
def config():
    """
    Load the YAML configuration for testing.
    """
    return read_yaml_config(PATH_YAML)


@pytest.fixture(scope="module")
def sample_data(spark):
    """
    Sample PySpark DataFrame for testing.
    """
    data = [
        {
            "asin": "123B",
            "title": "Product 1",
            "description": "This is a great product.",
            "feature": "Fast",
            "brand": "BrandA",
            "main_cat": "Electronics",
        },
        {
            "asin": "123C",
            "title": "Product 2",
            "description": None,
            "feature": "Cheap",
            "brand": "BrandB",
            "main_cat": "Home",
        },
        {
            "asin": "123D",
            "title": "Product 3",
            "description": "Works well",
            "feature": None,
            "brand": "BrandC",
            "main_cat": "Outdoors",
        },
    ]
    return spark.createDataFrame(data)


def test_load_config(config):
    """
    Test loading the YAML configuration file.
    """
    assert "etl_pipeline" in config
    assert "tokenizer" in config
    assert config["etl_pipeline"]["data_fraction"] == 1.0


def test_load_tokenizer(config):
    """
    Test loading the tokenizer using the configuration.
    """
    tokenizer_model = config["etl_pipeline"]["tokenizer_model"]
    tokenizer = load_tokenizer(tokenizer_model)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)


def test_tokenize_and_transform(config, sample_data):
    """
    Test tokenization using the tokenizer and config parameters.
    """
    tokenizer_model = config["etl_pipeline"]["tokenizer_model"]
    tokenizer = load_tokenizer(tokenizer_model)

    # Add combined_text column for tokenization
    sample_data = sample_data.withColumn("combined_text", sample_data.title)

    tokenized_df = tokenize_and_transform(sample_data, tokenizer, config["tokenizer"])
    assert "input_ids" in tokenized_df.columns
    assert "attention_mask" in tokenized_df.columns


def test_etl_pipeline(mocker):
    """
    Test the end-to-end ETL pipeline using mocked input and output paths.
    """
    # Mocking the I/O operations
    mocker.patch("src.data.utils.load_jsonl_gz", return_value=None)  # Mock extraction
    mocker.patch("src.data.etl.save_to_parquet")  # Mock saving

    # Run pipeline with config
    etl_pipeline(PATH_YAML)
