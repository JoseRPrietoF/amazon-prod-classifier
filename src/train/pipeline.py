import datetime
import mlflow
import fire
import torch
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from sklearn.preprocessing import LabelEncoder
from src.train import utils
from src.train.utils import CATEGORIES
from src.train.datasets.data_manager import DataManager
from src.train.trainers.trainer import Trainer

logger = utils.setup_logger(__name__)


def run(config="src/training/configs/default_config.yml", **overrides):
    """
    Function to run the training pipeline.

    Args:
        config (str): Path to the configuration file. Defaults to 'src/training/default_config.yml'.
            input_data_path (str): Path to the input JSONL.gz file.
            input_data_encoded (bool): Whether the data is already encoded. Defaults to False.
            bert_model_name (str): Name of the BERT model to use for tokenization. Defaults to 'bert-base-uncased'.
            trainable_layers (int | None): Number of layers to keep trainable. None for all layers. Defaults to None.
            num_epochs (int): Number of epochs to train the model. Defaults to 3.
            batch_size (int): Batch size for training and validation. Defaults to 8.
            test_batch_size (int): Batch size for testing. Defaults to 32.
            learning_rate (float): Learning rate for the optimizer. Defaults to 1e-5.
            optimizer (str): Optimizer to use for training. Defaults to 'AdamW'.
            seed (int): Random seed for reproducibility. Defaults to 42.
            device (str): Device to use for training. Defaults to 'cuda'.
            data_fraction (float): Fraction of the data to sample for training. Defaults to 1.0.
    """
    try:
        logger.info("Logging in to MLflow server hosted at Databricks CE.")
        # mlflow.set_tracking_uri(SETTINGS["MLFLOW_TRACKING_URI"])

        logger.info("Setting up MLflow experiment.")
        # mlflow.set_experiment(SETTINGS["MLFLOW_EXPERIMENT_NAME"])

        config = utils.read_yaml_config(config)
        config.update(overrides)

        # Extract configuration parameters
        input_data_path = config.get(
            "input_data_path",
            "data/processed/amz_products_small_processed_bert_tiny_v1.parquet",
        )
        bert_model_name = config.get("bert_model_name", "bert-base-uncased")
        input_data_encoded = config.get("input_data_encoded", False)
        trainable_layers = config.get("trainable_layers", None)
        max_length = config.get("max_length", 512)
        num_epochs = config.get("num_epochs", 3)
        batch_size = config.get("batch_size", 8)
        test_batch_size = config.get("test_batch_size", 32)
        learning_rate = float(config.get("learning_rate", 1e-5))
        seed = config.get("seed", 42)
        device = config.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        data_fraction = config.get("data_fraction", 1.0)

        # Set random seed for reproducibility
        logger.info("Setting random seed: %d.", seed)
        utils.set_random_seed(seed, device=device)

        # Start MLflow tracking
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        with mlflow.start_run(run_name=f"Run {current_time}"):
            mlflow.log_params(config)

            # Encode categories
            label_encoder = LabelEncoder()
            label_encoder.fit(CATEGORIES)

            # Initialize tokenizer and model
            logger.info("Loading BERT tokenizer and model: %s.", bert_model_name)
            tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                bert_model_name, num_labels=len(CATEGORIES)
            )
            model.to(device)

            # Freeze layers if specified
            if trainable_layers is not None and trainable_layers != "None":
                logger.info(
                    "Freezing all layers except the last %d layers.", trainable_layers
                )
                total_params, trainable_params = utils.freeze_model_layers(
                    model, int(trainable_layers)
                )
                logger.info(
                    "Total model parameters: %d, Trainable parameters: %d.",
                    total_params,
                    trainable_params,
                )

            # Load data and create datasets/loaders
            logger.info("Loading and preparing data.")
            df = utils.load_parquet(input_data_path)
            if data_fraction < 1.0:
                df = df.sample(fraction=data_fraction, seed=seed)

            train_dataset, val_dataset, test_dataset = DataManager.create_datasets(
                df,
                label_encoder,
                tokenizer,
                input_data_encoded,
                max_length=max_length,
                seed=seed,
            )
            train_loader, val_loader, test_loader = DataManager.create_loaders(
                train_dataset, val_dataset, test_dataset, batch_size, test_batch_size
            )

            # Initialize optimizer and scheduler
            optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
            total_steps = len(train_loader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps,
            )

            # Initialize Trainer
            trainer = Trainer(
                model, optimizer=optimizer, scheduler=scheduler, device=device
            )

            # Training and Validation Loop
            trainer.train_and_evaluate(train_loader, val_loader, num_epochs, seed)

            # Testing
            decoded_categories = label_encoder.inverse_transform(
                label_encoder.transform(CATEGORIES)
            )
            trainer.test(test_loader, decoded_categories)

    except Exception as e:
        logger.error(
            "An error occurred during the training pipeline: %s", str(e), exc_info=True
        )
        raise


if __name__ == "__main__":
    fire.Fire(run)
