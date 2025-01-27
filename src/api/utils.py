"""Module for Pydantic schema used in the API for input data."""

from pydantic import BaseModel, create_model
from enum import Enum
from src.train.utils import CATEGORIES
from src.data.utils import read_yaml_config
from typing import Any, Dict

SETTINGS = read_yaml_config("src/api/config.yaml")


class GetDataRequest(BaseModel):
    """Pydantic model for requesting a specific number of items."""

    num_items: int


"""Used for Pydantic schema used in the API for predictions response."""

CategoryEnum = Enum(
    "CategoryEnum",
    {c.replace("&", "and").replace(", ", "_").replace(" ", "_"): c for c in CATEGORIES},
)


class PredictionResponse(BaseModel):
    """
    Pydantic schema for the predictions response.

    Attributes:
        main_cat (CategoryEnum): The predicted category.
    """

    main_cat: CategoryEnum


# def preprocess_data(data: BaseModel) -> str:
#     """
#     Preprocesses the input data for the model.

#     Args:
#         data (ProductDescription): The input data to preprocess.

#     Returns:
#         str: The preprocessed text data.
#     """
#     combined_text = " | ".join([data.asin] +
#                               [data.title] + data.description + data.feature + [data.brand])
#     return combined_text


def preprocess_data(data: BaseModel) -> str:
    """
    Preprocesses the input data for the model dynamically.

    Args:
        data (BaseModel): The input data to preprocess.

    Returns:
        str: The preprocessed text data.
    """
    combined_text = " | ".join(
        str(getattr(data, field, ""))  # Access each attribute dynamically
        for field in data.model_fields  # Iterate over all fields in the Pydantic model
        if field != "image"
        and getattr(
            data, field, None
        )  # Skip "image" and check if the attribute is not None
    )
    return combined_text


def create_dynamic_model(schema: Dict[str, Any], model_name: str) -> BaseModel:
    """
    Creates a dynamic Pydantic model based on the schema.

    Args:
        schema (Dict[str, Any]): Schema definition.
        model_name (str): Name of the dynamic model.

    Returns:
        BaseModel: A dynamically created Pydantic model.
    """
    fields = {
        key: (eval(value.split("[")[0]), ...)
        if "List" not in value
        else (eval(value), ...)
        for key, value in schema.items()
    }
    return create_model(model_name, **fields)
