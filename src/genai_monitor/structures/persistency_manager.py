import json
from pathlib import Path
from typing import Dict, Optional, Union

from loguru import logger

from genai_monitor.common.structures.data import Artifact, Conditioning, Sample
from genai_monitor.config import PersistencyManagerConfig
from genai_monitor.static.constants import DEFAULT_PERSISTENCY_PATH


class PersistencyManager:
    """Manager for saving and loading model outputs in binary format to disk."""

    path: Path
    enabled: bool = False
    _configured: bool = False

    def __init__(self, path: Union[str, Path], enabled: bool):  # noqa: D107, ANN204
        self.enabled = enabled
        path = path if path is not None else DEFAULT_PERSISTENCY_PATH

        if self.enabled:
            self.path = Path(path) if isinstance(path, str) else path
            self.path.mkdir(parents=True, exist_ok=True)

        self._configured = True
        status = "enabled" if self.enabled else "disabled"

        logger.info(f"PersistencyManager configured in {status} mode with path: {self.path}.")

    def configure(self, config: PersistencyManagerConfig):
        """Configures the persistency manager.

        Args:
            config: The configuration object specifying the parameters of PersistencyManager.

        Raises:
            ValueError: If the configuration file cannot be loaded.
        """
        self.enabled = config.enabled
        path = config.path if config.path is not None else DEFAULT_PERSISTENCY_PATH

        if self.enabled:
            self.path = Path(path) if isinstance(path, str) else path
            self.path.mkdir(parents=True, exist_ok=True)

        self._configured = True
        logger.info(
            f"PersistencyManager configured in {'enabled' if self.enabled else 'disabled'} mode with path: {self.path}."
        )

    def save_sample(self, sample: Sample):
        """Saves the sample data to a binary file.

        Args:
            sample: The sample to save.

        Raises:
            ValueError: If the sample data is None.
            ValueError: If the sample ID is None.
            ValueError: If the PersistencyManager is disabled.
        """
        if not self.is_configured():
            raise ValueError("PersistencyManager has not been configured.")

        if self.enabled:
            if sample.data is None:
                raise ValueError("Sample data is None.")

            if sample.id is None:
                raise ValueError("Sample has to be written to the database before saving.")

            self.save_bytes_to_disk(sample.data, f"samples/{sample.id}.bin")
            logger.success(f"Saved sample #{sample.id} to disk.")

    def load_sample(self, sample: Sample) -> bytes:
        """Loads the binary file for the given sample.

        Args:
            sample: The sample to load.

        Returns:
            The binary data from the file.

        Raises:
            ValueError: If the PersistencyManager is disabled.
            ValueError: If the sample ID is None.
            FileNotFoundError: If the binary file for the sample ID is not found.
            ValueError: If the PersistencyManager has not been configured.
        """
        if not self.is_configured():
            raise ValueError("PersistencyManager has not been configured.")

        if not self.enabled:
            raise ValueError("Cannot load binary file while PersistencyManager is disabled.")

        if sample.id is None:
            raise ValueError("Sample id is None.")

        bytesdata = self.load_bytes_from_disk(f"samples/{sample.id}.bin")
        logger.success(f"Loaded sample #{sample.id} from disk.")
        return bytesdata

    def save_bytes_to_disk(self, data: bytes, filename: str):
        """Saves the bytes to a binary file.

        Args:
            data: The bytes to save.
            filename: The name of the file to save the bytes to.

        Raises:
            ValueError: If the PersistencyManager is disabled.
        """
        if not self.is_configured():
            raise ValueError("PersistencyManager has not been configured.")

        if self.enabled:
            filepath = self.path / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as file:
                file.write(data)

    def load_bytes_from_disk(self, filename: str) -> bytes:
        """Loads the bytes from a binary file.

        Args:
            filename: The name of the file to load the bytes from.

        Returns:
            The bytes from the file.

        Raises:
            FileNotFoundError: If the file is not found.
        """
        with open(self.path / filename, "rb") as file:
            bytesdata = file.read()
        return bytesdata

    def is_configured(self) -> bool:
        """Checks if the PersistencyManager has been configured.

        Returns:
            Boolean indicating if the PersistencyManager has been configured.
        """
        return self._configured

    def save_artifact(self, artifact: Artifact):
        """Saves the artifact data to a binary file.

        Args:
            artifact: The artifact to save.

        Raises:
            ValueError: If the artifact ID is None.
            ValueError: If the PersistencyManager is disabled.
        """
        if not self.is_configured():
            raise ValueError("PersistencyManager has not been configured.")

        if self.enabled:
            if artifact.id is None:
                raise ValueError("Artifact has to be written to the database before saving.")

            self.save_bytes_to_disk(json.dumps(artifact.value).encode("utf-8"), f"artifacts/{artifact.id}.bin")
            logger.success(f"Saved artifact #{artifact.id} to disk.")

    def load_artifact(self, artifact: Artifact) -> bytes:
        """Loads the binary file for the given artifact.

        Args:
            artifact: The artifact to load.

        Returns:
            The binary data from the file.

        Raises:
            ValueError: If the PersistencyManager is disabled.
            ValueError: If the artifact ID is None.
            FileNotFoundError: If the binary file for the artifact ID is not found.
            ValueError: If the PersistencyManager has not been configured.
        """
        if not self.is_configured():
            raise ValueError("PersistencyManager has not been configured.")

        if not self.enabled:
            raise ValueError("Cannot load binary file while PersistencyManager is disabled.")

        if artifact.id is None:
            raise ValueError("Artifact id is None.")

        bytesdata = self.load_bytes_from_disk(f"artifacts/{artifact.id}.bin")
        logger.success(f"Loaded artifact #{artifact.id} from disk.")
        return json.loads(bytesdata.decode("utf-8"))

    def save_conditioning(self, conditioning: Conditioning):
        """Saves the conditioning data to a binary file.

        Args:
            conditioning: The conditioning to save.

        Raises:
            ValueError: If the conditioning ID is None.
            ValueError: If the PersistencyManager is disabled.
        """
        if not self.is_configured():
            raise ValueError("PersistencyManager has not been configured.")

        if self.enabled:
            if conditioning.id is None:
                raise ValueError("Conditioning has to be written to the database before saving.")

            self.save_bytes_to_disk(
                json.dumps(conditioning.value).encode("utf-8"), f"conditionings/{conditioning.id}.bin"
            )
            logger.success(f"Saved conditioning #{conditioning.id} to disk.")

    def load_conditioning(self, conditioning: Conditioning) -> Dict:
        """Loads the binary file for the given conditioning.

        Args:
            conditioning: The conditioning to load.

        Returns:
            The binary data from the file.

        Raises:
            ValueError: If the PersistencyManager is disabled.
            ValueError: If the conditioning ID is None.
            FileNotFoundError: If the binary file for the conditioning ID is not found.
            ValueError: If the PersistencyManager has not been configured.
        """
        if not self.is_configured():
            raise ValueError("PersistencyManager has not been configured.")

        if not self.enabled:
            raise ValueError("Cannot load binary file while PersistencyManager is disabled.")

        if conditioning.id is None:
            raise ValueError("Conditioning id is None.")

        bytesdata = self.load_bytes_from_disk(f"conditionings/{conditioning.id}.bin")
        logger.success(f"Loaded conditioning #{conditioning.id} from disk.")
        return json.loads(bytesdata.decode("utf-8"))
