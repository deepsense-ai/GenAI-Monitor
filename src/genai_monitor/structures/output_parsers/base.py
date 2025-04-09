import hashlib
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from loguru import logger

from genai_monitor.common.structures.data import Sample
from genai_monitor.utils.data_hashing import BaseType, hash_base_type

T = TypeVar("T")


class BaseModelOutputParser(ABC, Generic[T]):
    """Abstract class for converting between sample and model output types."""

    _known_classes: List[str] = []

    @abstractmethod
    def model_output_to_bytes(self, model_output: T) -> bytes:
        """Converts the model output to a byte representation.

        Parameters:
            model_output: The model output to convert.

        Returns:
            The byte representation of the model output.
        """

    @abstractmethod
    def bytes_to_model_output(self, databytes: bytes) -> T:
        """Converts a byte representation back into model output.

        Parameters:
            databytes: The byte representation of the model output.

        Returns:
            The model output reconstructed from the byte representation.
        """

    def get_model_output_hash(self, data: T) -> str:
        """Calculates the hash value of the given data.

        Parameters:
            data: The data to calculate the hash value for.

        Returns:
            The hash value of the data.
        """
        try:
            return self.get_base_type_hash(model_output=data)
        except NotImplementedError:
            logger.info(
                "Conversion of model output to base type not implemented, falling back to calculation of the full hash."
            )
        data_bytes = self.model_output_to_bytes(data)
        return hashlib.sha256(data_bytes).hexdigest()

    def get_base_type_hash(self, model_output: T) -> str:
        """Calculate the hash of the base data type stored in the model output.

        Args:
            model_output: The model outuput to extract the base type from.

        Returns:
            The data in model output as one of the base supported types.
        """
        base_type_data = self.model_output_to_base_type(model_output=model_output)
        return hash_base_type(data=base_type_data)

    def model_output_to_base_type(self, model_output: T) -> BaseType:
        """Get the base type to calculate the hash upon.

        Args:
            model_output: The output of the model to extract the data from

        Returns:
            One of the base data types supported in the system
        """
        raise NotImplementedError("Method to extract base types from model output not specified.")

    def get_sample_from_model_output(self, model_output: T) -> Sample:
        """Converts the model output to a sample.

        Parameters:
            model_output: The model output to convert to a sample.

        Returns:
            The sample created from the model output.
        """
        data_bytes = self.model_output_to_bytes(model_output)
        model_output_hash = self.get_model_output_hash(model_output)

        return Sample(data=data_bytes, hash=model_output_hash)

    def get_model_output_from_sample(self, sample: Sample) -> T:
        """Converts the sample to model output.

        Parameters:
            sample: The sample to convert to model output.

        Returns:
            The model output created from the sample.

        Raises:
            ValueError: if the data of a sample is empty
        """
        if sample.data is None:
            raise ValueError("Sample data is empty.")

        return self.bytes_to_model_output(sample.data)
