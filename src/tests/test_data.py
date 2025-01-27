import pytest
from pyspark.sql import SparkSession
from src.data.preprocessing import PreprocessingPipeline


# Fixture for SparkSession
@pytest.fixture(scope="module")
def spark():
    return (
        SparkSession.builder.appName("TestPreprocessingPipeline")
        .master("local[*]")
        .getOrCreate()
    )


# Fixture for a sample DataFrame
@pytest.fixture
def sample_df(spark):
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


def test_validate_input_correct(sample_df):
    """
    Test that validate_pipeline_input passes with correct data.
    """
    pipeline = PreprocessingPipeline()
    # Should not raise any exception
    pipeline.validate_pipeline_input(sample_df)


def test_validate_input_empty(spark):
    """
    Test that validate_pipeline_input raises error for empty DataFrame.
    """
    pipeline = PreprocessingPipeline()
    empty_df = spark.createDataFrame(
        [],
        schema="title STRING, description STRING, feature STRING, brand STRING, main_cat STRING",
    )

    with pytest.raises(ValueError, match="Input DataFrame is empty or None."):
        pipeline.validate_pipeline_input(empty_df)


def test_validate_input_missing_columns(spark):
    """
    Test that validate_pipeline_input raises error for missing columns.
    """
    pipeline = PreprocessingPipeline()
    incomplete_df = spark.createDataFrame(
        [{"title": "Product 1", "main_cat": "Electronics"}]
    )

    with pytest.raises(ValueError, match="Input DataFrame is missing required columns"):
        pipeline.validate_pipeline_input(incomplete_df)


def test_clean_data(sample_df):
    """
    Test that _clean_data retains only the necessary columns.
    """
    pipeline = PreprocessingPipeline()
    cleaned_df = pipeline._clean_data(sample_df)

    # Check columns
    assert set(cleaned_df.columns) == set(
        pipeline.text_columns + [pipeline.label_column]
    )


def test_combine_text_fields(sample_df):
    """
    Test that _combine_text_fields correctly combines text fields.
    """
    pipeline = PreprocessingPipeline()
    combined_df = pipeline._combine_text_fields(sample_df)

    # Check if combined text column exists
    assert pipeline.combined_text_column in combined_df.columns

    # Check combined values for the first row
    row = combined_df.filter(combined_df.title == "Product 1").first()
    print(row.combined_text)
    assert (
        row.combined_text
        == "123B | Product 1 | This is a great product. | Fast | BrandA"
    )


def test_preprocess_pipeline(sample_df):
    """
    Test that preprocess runs the full pipeline and produces expected output.
    """
    pipeline = PreprocessingPipeline()
    result_df = pipeline.preprocess(sample_df)

    # Check columns in final DataFrame
    assert set(result_df.columns) == {
        pipeline.combined_text_column,
        pipeline.label_column,
    }

    # Check if output row count matches input
    assert result_df.count() == sample_df.count()

    # Check combined text values
    row = result_df.filter(result_df.main_cat == "Electronics").first()
    assert (
        row.combined_text
        == "123B | Product 1 | This is a great product. | Fast | BrandA"
    )
