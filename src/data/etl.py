"""Module that performs the ETL pipeline (Extract, Transform, Load)."""

import fire
from transformers import AutoTokenizer
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, IntegerType
from src.data.utils import load_jsonl_gz, setup_logger, read_yaml_config
from src.data.preprocessing import PreprocessingPipeline
from typing import Dict

logger = setup_logger(__name__)


def etl_pipeline(config_path: str = "config.yaml"):
    """
    Full ETL pipeline using parameters from a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file. Defaults to "config.yaml".
    """
    try:
        # Load configuration
        config = read_yaml_config(config_path)

        # Extract pipeline parameters
        input_path = config["etl_pipeline"]["input_path"]
        output_path = config["etl_pipeline"]["output_path"]
        data_fraction = config["etl_pipeline"]["data_fraction"]
        seed = config["etl_pipeline"]["seed"]
        tokenizer_model = config["etl_pipeline"].get("tokenizer_model")

        # Tokenizer parameters
        tokenizer_params = config["tokenizer"]

        logger.info("Starting ETL pipeline.")

        # Step 1: Extract
        logger.info("Extracting raw data from: %s", input_path)
        raw_df = load_jsonl_gz(input_path, data_fraction, seed)

        # Step 2: Transform
        logger.info("Transforming data.")
        bert_tokenizer = None
        if tokenizer_model:
            bert_tokenizer = load_tokenizer(tokenizer_model)
        transformed_df = preprocess_data(
            raw_df, bert_tokenizer, tokenizer_params, config["etl_pipeline"]
        )

        # Step 3: Load
        logger.info("Loading transformed data to: %s", output_path)
        save_to_parquet(transformed_df, output_path)

        logger.info("ETL pipeline completed successfully.")
    except Exception as e:
        logger.error("ETL pipeline failed: %s", e, exc_info=True)
        raise


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Loads the specified BERT tokenizer.

    Args:
        model_name (str): Name of the BERT tokenizer model.

    Returns:
        AutoTokenizer: The loaded tokenizer.
    """
    try:
        logger.info("Loading BERT tokenizer: %s", model_name)
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error("Failed to load tokenizer: %s", e, exc_info=True)
        raise


def preprocess_data(
    df: DataFrame,
    tokenizer: AutoTokenizer = None,
    tokenizer_params=None,
    etl_config: Dict = None,
) -> DataFrame:
    """
    Preprocesses raw data and applies BERT tokenization if a tokenizer is provided.

    Args:
        df (DataFrame): Input DataFrame with raw product data.
        tokenizer (AutoTokenizer, optional): Tokenizer for text processing. Defaults to None.
        tokenizer_params (dict, optional): Parameters for the tokenizer. Defaults to None.

    Returns:
        DataFrame: Transformed DataFrame ready for loading.
    """
    try:
        logger.info("Initializing preprocessing pipeline.")
        pipeline = PreprocessingPipeline(
            label_column=etl_config["label_column"],
            text_columns=etl_config["text_columns"],
        )
        processed_df = pipeline.preprocess(df)

        if tokenizer:
            logger.info(f"Applying BERT tokenization with {tokenizer.name_or_path}.")
            processed_df = tokenize_and_transform(
                processed_df, tokenizer, tokenizer_params
            )

        return processed_df
    except Exception as e:
        logger.error("Preprocessing failed: %s", e, exc_info=True)
        raise


def tokenize_and_transform(
    df: DataFrame, tokenizer: AutoTokenizer, tokenizer_params: dict
) -> DataFrame:
    """
    Tokenizes and transforms text data using the BERT tokenizer with configurable parameters.

    Args:
        df (DataFrame): Input DataFrame with preprocessed text.
        tokenizer (AutoTokenizer): BERT tokenizer.
        tokenizer_params (dict): Tokenizer parameters.

    Returns:
        DataFrame: Transformed DataFrame with tokenized text columns.
    """
    try:
        # Define the tokenization function for UDF
        def tokenize(text):
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=tokenizer_params["max_length"],
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
            )
            return [encoding["input_ids"], encoding["attention_mask"]]

        # UDF to tokenize text columns
        tokenize_udf = udf(tokenize, ArrayType(ArrayType(IntegerType())))

        # Apply the tokenizer
        df = df.withColumn("tokens", tokenize_udf(col("combined_text")))
        df = df.withColumn("input_ids", df["tokens"].getItem(0))
        df = df.withColumn("attention_mask", df["tokens"].getItem(1))

        # Drop intermediate columns
        df = df.drop("combined_text")
        df = df.drop("tokens")
        return df
    except Exception as e:
        logger.error("Tokenization failed: %s", e, exc_info=True)
        raise


def save_to_parquet(df: DataFrame, output_path: str):
    """
    Saves the DataFrame to a Parquet file.

    Args:
        df (DataFrame): DataFrame to save.
        output_path (str): Path where the Parquet file will be saved.
    """
    try:
        df.write.mode("overwrite").parquet(output_path)
        logger.info("Data saved successfully to: %s", output_path)
    except Exception as e:
        logger.error("Failed to save data to Parquet: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    fire.Fire(etl_pipeline)
