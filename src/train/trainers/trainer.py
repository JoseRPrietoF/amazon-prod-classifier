import numpy as np
import mlflow
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
from src.data.utils import setup_logger
from src.train.utils import plot_confusion_matrix

logger = setup_logger(__name__)


class Trainer:
    """Custom training, evaluation, and testing loop for sequence classification tasks."""

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: str,
    ):
        """
        Initializes the trainer with the model, optimizer, scheduler, and device.
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _move_to_device(self, batch):
        """
        Moves input data to the specified device.
        """
        input_ids, attention_mask, labels = batch
        return (
            input_ids.to(self.device),
            attention_mask.to(self.device),
            labels.to(self.device),
        )

    def _calculate_metrics(self, preds, labels):
        """
        Calculates classification metrics.
        """
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_score": f1_score(labels, preds, average="weighted"),
            "precision": precision_score(
                labels, preds, average="weighted", zero_division=0
            ),
            "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        }

    def train_epoch(self, loader: DataLoader, epoch: int, num_epochs: int):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(loader, desc=f"Training Epoch {epoch}/{num_epochs}"):
            self.optimizer.zero_grad()
            input_ids, attention_mask, labels = self._move_to_device(batch)

            outputs = self.model(
                input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.cpu().numpy()
            total_loss += loss.item()
            total_correct += (logits.argmax(axis=1) == label_ids).sum()
            total_samples += len(label_ids)

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_samples

        logger.info(
            f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4%}"
        )
        mlflow.log_metric("Train Loss", avg_loss, step=epoch)
        mlflow.log_metric("Train Accuracy", accuracy, step=epoch)

    def evaluate(self, loader: DataLoader, epoch: int, split: str = "Validation"):
        """
        Evaluates the model on a given dataset.
        """
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{split} Epoch {epoch}"):
                input_ids, attention_mask, labels = self._move_to_device(batch)

                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                total_loss += outputs.loss.item()

                logits = outputs.logits.detach().cpu().numpy()
                all_preds.extend(np.argmax(logits, axis=1).tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(loader)
        metrics = self._calculate_metrics(all_preds, all_labels)
        metrics["avg_loss"] = avg_loss
        metrics["all_labels"] = all_labels
        metrics["all_preds"] = all_preds

        logger.info(
            f"{split} - Loss: {avg_loss:.4f}, Accuracy: {metrics['accuracy']:.4%}"
        )
        mlflow.log_metrics(
            {f"{split} Loss": avg_loss, f"{split} Accuracy": metrics["accuracy"]},
            step=epoch,
        )
        return metrics

    def test(self, loader: DataLoader, decoded_categories: list[str]):
        """
        Tests the model on the test set and logs metrics.
        """
        metrics = self.evaluate(loader, epoch=0, split="Test")
        cm = confusion_matrix(metrics["all_labels"], metrics["all_preds"])
        plot_confusion_matrix(cm, decoded_categories)
        metrics.pop("avg_loss", None)
        metrics.pop("all_labels", None)
        metrics.pop("all_preds", None)
        logger.info(f"Test Metrics: {metrics}")

    def train_and_evaluate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        seed: int,
    ):
        """
        Handles the complete training and validation process, including saving the best model.
        """
        best_model_state = self.model.state_dict()
        best_eval_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            self.train_epoch(train_loader, epoch, num_epochs)
            eval_metrics = self.evaluate(val_loader, epoch, split="Validation")

            # Save the best model based on validation loss
            if eval_metrics["avg_loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["avg_loss"]
                best_model_state = self.model.state_dict()
                logger.info(
                    f"New best model saved at epoch {epoch} with Validation Loss: {eval_metrics['avg_loss']:.4f}."
                )

        # Restore the best model
        self.model.load_state_dict(best_model_state)
        logger.info("Best model loaded for testing.")
        # Log the best model
        mlflow.pytorch.log_model(self.model, "model")
