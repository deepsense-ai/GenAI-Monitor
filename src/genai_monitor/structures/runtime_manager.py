from collections import deque
from typing import Deque, Optional

from attrs import define, field

from genai_monitor.common.structures.data import Artifact, Sample


@define
class RuntimeManager:
    """Class for managing runtime data.

    Attributes:
        user_id: The user ID.
        version: The runtime version identifier.
        latest_sample: The most recently created sample.
        artifacts_for_next_sample: Collection of artifacts to be associated with the next sample.
    """

    user_id: Optional[int] = None
    version: Optional[str] = None
    latest_sample: Optional[Sample] = None
    artifacts_for_next_sample: Deque[Artifact] = field(factory=deque)

    def set_user_id(self, user_id: int) -> None:
        """Set the user ID.

        Args:
            user_id: The user ID.
        """
        self.user_id = user_id

    def set_runtime_version(self, version: str) -> None:
        """Set the version.

        Args:
            version: The version.
        """
        self.version = version
