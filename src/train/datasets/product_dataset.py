"""Module defining PyTorch Datasets for product data."""

from typing import Iterator
from torch import Tensor, tensor
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase
from pyspark.sql import DataFrame
from pyspark.sql.functions import rand
from sklearn.preprocessing import LabelEncoder
from src.train import utils

logger = utils.setup_logger(__name__)


class ProductIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for streaming product data from a Spark DataFrame.

    Attributes:
        df (DataFrame): Spark DataFrame containing product data.
        label_encoder (LabelEncoder): Encoder for category labels.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for text fields.
        is_encoded (bool): If True, assumes data is pre-encoded.
        max_length (int): Maximum sequence length for tokenization. Defaults to 512.
    """

    def __init__(
        self,
        df: DataFrame,
        label_encoder: LabelEncoder,
        tokenizer: PreTrainedTokenizerBase,
        is_encoded: bool,
        max_length=512,
    ):
        """
        Initialize the dataset.

        Args:
            df (DataFrame): Spark DataFrame with product data.
            label_encoder (LabelEncoder): Encoder for converting category labels.
            tokenizer (PreTrainedTokenizerBase): Tokenizer for text fields.
            is_encoded (bool): If True, assumes data is already tokenized.
            max_length (int): Maximum length for tokenization. Defaults to 512.
        """
        if not isinstance(df, DataFrame):
            raise ValueError("`df` must be a PySpark DataFrame.")
        if not isinstance(label_encoder, LabelEncoder):
            raise ValueError("`label_encoder` must be an instance of LabelEncoder.")
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError("`tokenizer` must be an instance of AutoTokenizer.")

        self.df = df
        self.label_encoder = label_encoder
        self.tokenizer = tokenizer
        self.is_encoded = is_encoded
        self.max_length = max_length

    def reshuffle(self, seed=42):
        """Reshuffles the Spark DataFrame."""
        self.df = self.df.orderBy(rand(seed=seed))

    def _encode_label(self, category: str) -> Tensor:
        """Encodes a category label into a tensor."""
        return tensor(self.label_encoder.transform([category])[0])

    def _tokenize(self, text: str) -> tuple[Tensor, Tensor]:
        """Tokenizes text using the BERT tokenizer."""
        encoded_input = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoded_input["input_ids"].squeeze(0), encoded_input[
            "attention_mask"
        ].squeeze(0)

    def _process_row(self, row, is_encoded: bool) -> tuple[Tensor, Tensor, Tensor]:
        """
        Processes a single row of the DataFrame.

        Args:
            row: A single row of data.
            is_encoded (bool): If True, assumes data is pre-encoded.

        Returns:
            tuple[Tensor, Tensor, Tensor]: input_ids, attention_mask, and label tensors.
        """
        try:
            label = self._encode_label(row["main_cat"])
            if is_encoded:
                input_ids = tensor(row["input_ids"])
                attention_mask = tensor(row["attention_mask"])
            else:
                input_ids, attention_mask = self._tokenize(row["combined_text"])
            return input_ids, attention_mask, label
        except Exception as e:
            logger.error(f"Failed to process row: {row}. Error: {e}", exc_info=True)
            raise

    def __len__(self):
        return self.df.count()

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        """
        Yield batches of data as tuples of input_ids, attention_mask, and labels.

        Yields:
            tuple[Tensor, Tensor, Tensor]: Batched input_ids, attention_mask, and labels.
        """
        return (
            self._process_row(row, self.is_encoded)
            for row in self.df.rdd.toLocalIterator()
        )
