"""This module contains the FastAPI application and its routes."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.pytorch
from transformers import AutoTokenizer
import torch
from src.train.utils import CATEGORIES
from src.api.utils import (
    PredictionResponse,
    preprocess_data,
    GetDataRequest,
    create_dynamic_model,
)  # ProductDescription
from src.data.utils import setup_logger, read_yaml_config
from src.data.utils import load_jsonl_gz
from pydantic import BaseModel
# from pyspark.sql.functions import rand


logger = setup_logger(__name__)

SETTINGS = read_yaml_config("src/api/config.yaml")

ml_artifacts = {}
# Load schema from YAML
schema = SETTINGS["ProductDescription"]

# Create a dynamic Pydantic model
ProductDescriptionDynamic = create_dynamic_model(schema, "ProductDescriptionDynamic")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.

    Handles initialization and teardown of global objects, such as loading ML models.
    """
    logger.info("Initializing application and loading MLflow artifacts.")

    mlflow.set_tracking_uri(SETTINGS["MLFLOW_TRACKING_URI"])
    # mlflow.set_experiment(SETTINGS["MLFLOW_EXPERIMENT_NAME"])

    # Get the parameter 'bert_model_name' from the run
    try:
        # Configure MLflow
        run_id = SETTINGS.get("MLFLOW_RUN_ID")
        if not run_id:
            raise ValueError(
                "MLFLOW_RUN_ID is not specified in the environment settings."
            )

        # Load the PyTorch model from MLflow
        model_path = f"runs:/{run_id}/model"
        print(SETTINGS["MLFLOW_TRACKING_URI"])
        logger.info("Loading model from MLflow at path: %s", model_path)
        model = mlflow.pytorch.load_model(model_path, map_location=torch.device("cpu"))

        # Retrieve the BERT model name from the MLflow run
        run = mlflow.get_run(run_id)
        bert_model_name = run.data.params.get("bert_model_name")
        if not bert_model_name:
            raise ValueError("BERT model name is not specified in the run parameters.")

        # Load tokenizer
        logger.info("Loading tokenizer for BERT model: %s", bert_model_name)
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

        # Store artifacts
        ml_artifacts["model"] = model
        ml_artifacts["tokenizer"] = tokenizer

        logger.info("MLflow artifacts loaded successfully.")
        logger.info("Loading example data.")
        df = load_jsonl_gz(
            SETTINGS["DATA_PATH_EXAMPLE"],
            float(SETTINGS["FRACTION_DATA"]),
            seed=int(SETTINGS["SEED"]),
        )
        ml_artifacts["data_example"] = df
        logger.info("Data loaded")
        yield
    except Exception as e:
        logger.error("Failed during application startup: %s", str(e))
        raise RuntimeError(f"Application initialization failed: {str(e)}")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def check_health():
    """
    Health check endpoint to ensure the API is running.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(product_data: BaseModel):
    """
    Predict the main category of a product based on its description.

    Args:
        product_data (ProductDescription): The input product data.

    """
    try:
        # Encode the categories
        # label_encoder could be loaded from Mlflow as an artifact attached to the run
        label_encoder = LabelEncoder()
        label_encoder.fit(CATEGORIES)
        # Retrieve the model and tokenizer from the ml_artifacts dictionary
        model = ml_artifacts["model"]
        tokenizer = ml_artifacts["tokenizer"]
        # Preprocess the input data

        preprocessed_text = preprocess_data(product_data)
        # encode the preprocessed text
        encoded_input = tokenizer.encode_plus(
            preprocessed_text,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=int(SETTINGS["MAX_LENGTH"]),
            truncation=True,
            padding="max_length",
        )
        # Run the model prediction
        with torch.no_grad():
            outputs = model(**encoded_input)
            prediction = torch.argmax(outputs.logits, dim=1)
        return PredictionResponse(
            main_cat=label_encoder.inverse_transform([prediction.item()])[0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/get_data", status_code=status.HTTP_200_OK)
async def get_data(request: GetDataRequest):
    """
    Retrieves a random sample of product data.

    Args:
        request (GetDataRequest): Number of items to retrieve.

    Returns:
        List[ProductDescription]: A list of randomly sampled products.
    """
    try:
        # Retrieve the PySpark DataFrame from ml_artifacts
        df = ml_artifacts.get("data_example")
        if df is None:
            raise RuntimeError("Example data is not loaded.")

        # Validate the requested number of items
        num_items = request.num_items
        if num_items <= 0:
            raise ValueError("Number of items must be greater than 0.")

        # Sample the required number of rows

        sampled_df = df.limit(num_items)  # orderBy(rand()) is too slow
        sampled_data = sampled_df.collect()
        # Convert the rows into ProductDescription objects
        schema
        product_descriptions = []

        for row in sampled_data:
            # Filtrar claves según las definidas en el schema
            filtered_row = {key: row[key] for key in schema if key in row}
            # Crear instancia del modelo dinámico
            product = ProductDescriptionDynamic(**filtered_row)
            print(f"Product: {product}")
            product_descriptions.append(product)
        # product_descriptions = [
        #     # ProductDescription(
        #     #     asin=row["asin"],
        #     #     brand=row["brand"],
        #     #     description=row["description"],
        #     #     feature=row["feature"],
        #     #     title=row["title"],
        #     #     image=row["image"]
        #     # )
        #     ProductDescriptionDynamic(row)
        #     for row in sampled_data
        # ]

        return product_descriptions
    except Exception as e:
        logger.error(f"Failed to retrieve data: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")
