"""Module for setting up the environment and utility functions."""

import random
import yaml
import torch
import numpy as np
import pandas as pd
from src.data.utils import load_parquet, setup_logger
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

logger = setup_logger(__name__)

CATEGORIES = [
    "All Electronics",
    "Amazon Fashion",
    "Amazon Home",
    "Arts, Crafts & Sewing",
    "Automotive",
    "Books",
    "Camera & Photo",
    "Cell Phones & Accessories",
    "Computers",
    "Digital Music",
    "Grocery",
    "Health & Personal Care",
    "Home Audio & Theater",
    "Industrial & Scientific",
    "Movies & TV",
    "Musical Instruments",
    "Office Products",
    "Pet Supplies",
    "Sports & Outdoors",
    "Tools & Home Improvement",
    "Toys & Games",
    "Video Games",
]


def parquet_to_df(path: str, use_pyspark: bool = False) -> pd.DataFrame:
    """
    Read a Parquet file into a Pandas DataFrame, optionally using PySpark.

    Args:
        path (str): Path to the Parquet file.
        use_pyspark (bool): If True, use PySpark to load the file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        if use_pyspark:
            return load_parquet(path).toPandas()
        return pd.read_parquet(path)
    except Exception as e:
        logger.error("Failed to read Parquet file: %s", e, exc_info=True)
        raise


def set_random_seed(seed: int, device: str = "cuda"):
    """
    Set random seed for reproducibility across libraries.

    Args:
        seed (int): Random seed value.
        device (str): Target device ('cuda' or 'cpu').
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def freeze_model_layers(model, layers_count: int):
    """
    Freeze all layers of a PyTorch model except the last `layers_count` layers.

    Args:
        model: PyTorch model.
        layers_count (int): Number of trainable layers to keep.

    Returns:
        tuple[int, int]: Total and trainable parameter counts.
    """
    layers = []
    if hasattr(model, "classifier"):
        layers.append(("classifier", model.classifier))
        layers_count -= 1
    if layers_count > 0 and hasattr(model.bert, "pooler"):
        layers.append(("bert.pooler", model.bert.pooler))
        layers_count -= 1
    if layers_count > 0:
        total_layers = len(model.bert.encoder.layer)
        layers_to_keep = min(layers_count, total_layers)
        start_index = total_layers - layers_to_keep
        layers.extend(
            [
                (f"bert.encoder.layer.{i}", layer)
                for i, layer in enumerate(
                    model.bert.encoder.layer[-layers_to_keep:], start=start_index
                )
            ]
        )

    for param in model.parameters():
        param.requires_grad = False
    trainable_params = 0
    for _, layer in layers:
        for param in layer.parameters():
            param.requires_grad = True
            trainable_params += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    return total_params, trainable_params


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], log: bool = True):
    """
    Plots and logs the confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix.
        labels (list[str]): List of class labels.
        log (bool): Whether to log the plot to MLflow. Defaults to True.
    """
    try:
        cm_percentage = cm * 100
        plt.figure(figsize=(20, 12))
        sns.heatmap(
            cm_percentage,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            annot_kws={"size": 10, "ha": "center", "va": "center"},
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual Labels")
        plt.xlabel("Predicted Labels")
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        # Adding an offset and aligning vertically
        plt.yticks(tick_marks + 0.5, labels, rotation=0, va="center")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()
        if log:
            mlflow.log_artifact("confusion_matrix.png")
    except Exception as e:
        logger.error(
            "Failed to plot or log confusion matrix: %s", str(e), exc_info=True
        )
        raise
