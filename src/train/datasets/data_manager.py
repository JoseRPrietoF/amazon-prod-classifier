from torch.utils.data import DataLoader
from pyspark.sql.functions import rand
from src.train.datasets.product_dataset import ProductIterableDataset


class DataManager:
    """Handles dataset creation and splitting."""

    @staticmethod
    def create_datasets(
        df, label_encoder, tokenizer, is_encoded=False, max_length=512, seed=42
    ):
        """
        Splits the data and creates PyTorch datasets.

        Args:
            df (DataFrame): Input Spark DataFrame.
            label_encoder (LabelEncoder): Encoder for labels.
            tokenizer (AutoTokenizer): Tokenizer for BERT.
            is_encoded (bool): Whether the data is already encoded.
            max_length (int): Max token sequence length. Defaults to 512.
            seed (int): Random seed for reproducibility. Defaults to 42.

        Returns:
            Tuple[ProductIterableDataset, ProductIterableDataset, ProductIterableDataset]: Train, val, test datasets.
        """
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)
        train_df, val_df = train_df.randomSplit([0.8, 0.2], seed=seed)

        datasets = {
            split: ProductIterableDataset(
                split_df.orderBy(rand(seed=seed)),
                label_encoder,
                tokenizer,
                is_encoded,
                max_length,
            )
            for split, split_df in zip(
                ["train", "val", "test"], [train_df, val_df, test_df]
            )
        }
        return datasets["train"], datasets["val"], datasets["test"]

    @staticmethod
    def create_loaders(train_ds, val_ds, test_ds, batch_size, test_batch_size):
        """
        Creates DataLoaders for train, val, and test datasets.

        Args:
            train_ds (ProductIterableDataset): Training dataset.
            val_ds (ProductIterableDataset): Validation dataset.
            test_ds (ProductIterableDataset): Test dataset.
            batch_size (int): Batch size for training and validation.
            test_batch_size (int): Batch size for testing.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, val, test DataLoaders.
        """
        return (
            DataLoader(train_ds, batch_size=batch_size),
            DataLoader(val_ds, batch_size=batch_size),
            DataLoader(test_ds, batch_size=test_batch_size),
        )
