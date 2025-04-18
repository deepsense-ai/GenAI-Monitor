from enum import Enum


class SampleStatus(Enum):
    """Enum representing the status of the sample."""

    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
