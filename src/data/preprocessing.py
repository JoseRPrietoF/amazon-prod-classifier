"""Module for preprocessing the product data for BERT-based NLP tasks."""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from src.data import utils

logger = utils.setup_logger(__name__)


class PreprocessingPipeline:
    """
    A reusable class to preprocess product data for BERT-based NLP tasks.

    Methods:
        preprocess(df: DataFrame) -> DataFrame:
            The entire pipeline including cleaning and text processing.
    """

    def __init__(
        self,
        text_columns=None,
        label_column="main_cat",
        combined_text_column="combined_text",
    ):
        """
        Initializes the preprocessing pipeline with the required configurations.

        Args:
            text_columns (list, optional): List of columns to combine into a single text field.
                If None, defaults to ["asin", "title", "description", "feature", "brand", "main_cat"].
            label_column (str, optional): Name of the label column. Defaults to "main_cat".
            combined_text_column (str, optional): Name of the combined text column. Defaults to "combined_text".
        """
        self.text_columns = text_columns or [
            "asin",
            "title",
            "description",
            "feature",
            "brand",
        ]
        self.label_column = label_column
        self.combined_text_column = combined_text_column

    def validate_pipeline_input(self, df: DataFrame):
        """
        Validates that the input DataFrame is not empty and contains required columns.

        Args:
            df (DataFrame): Input DataFrame to validate.

        Raises:
            ValueError: If the DataFrame is empty or missing required columns.
        """
        if df is None or df.rdd.isEmpty():
            raise ValueError("Input DataFrame is empty or None.")

        missing_columns = [
            col
            for col in self.text_columns + [self.label_column]
            if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Input DataFrame is missing required columns: {missing_columns}"
            )

    def _clean_data(self, df: DataFrame) -> DataFrame:
        """
        Cleans the input DataFrame by retaining only necessary columns and handling missing values.

        Args:
            df (DataFrame): Input DataFrame with raw product data.

        Returns:
            DataFrame: Cleaned DataFrame with only the necessary columns.
        """
        try:
            # Retain only the specified columns
            columns_to_keep = self.text_columns + [self.label_column]
            df = df.select(*[col for col in columns_to_keep if col in df.columns])
            return df
        except Exception as e:
            logger.error("Failed to clean data: %s", e, exc_info=True)
            raise

    def _combine_text_fields(self, df: DataFrame) -> DataFrame:
        """
        Combines specified text fields into a single column separated by a delimiter.

        Args:
            df (DataFrame): Input DataFrame with text columns.

        Returns:
            DataFrame: DataFrame with the combined text field added.
        """
        try:
            # Define a UDF to concatenate text fields with " | " as a delimiter
            combine_udf = udf(
                lambda *cols: " | ".join([str(col) for col in cols if col is not None]),
                StringType(),
            )

            # Add the combined text column
            return df.withColumn(
                self.combined_text_column,
                combine_udf(*[col(c) for c in self.text_columns if c in df.columns]),
            )
        except Exception as e:
            logger.error("Failed to combine text fields: %s", e, exc_info=True)
            raise

    def preprocess(self, df: DataFrame) -> DataFrame:
        """
        Executes the full preprocessing pipeline: validation, cleaning, and text processing.

        Args:
            df (DataFrame): Input DataFrame with raw product data.

        Returns:
            DataFrame: Preprocessed DataFrame ready for model training or inference.
        """
        try:
            logger.info("Validating input DataFrame.")
            self.validate_pipeline_input(df)

            logger.info("Starting preprocessing pipeline.")
            logger.info("Step 1: Cleaning the data.")
            cleaned_df = self._clean_data(df)

            logger.info("Step 2: Combining text fields.")
            processed_df = self._combine_text_fields(cleaned_df)

            logger.info("Step 3: Retaining only combined text and label columns.")
            final_df = processed_df.select(self.combined_text_column, self.label_column)

            logger.info("Preprocessing pipeline completed successfully.")
            return final_df
        except Exception as e:
            logger.error("Preprocessing pipeline failed: %s", e, exc_info=True)
            raise
