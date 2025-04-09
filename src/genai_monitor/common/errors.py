class NoRegisteredMethodsError(Exception):
    """Raised when attempting to wrap a class that has no registered methods in monitored_methods."""

    def __init__(self, cls_name: str):
        super().__init__(f"Class '{cls_name}' has no registered methods to wrap.")


class NoRegisteredParsersError(Exception):
    """Raised when attempting to wrap a class that does not have parser registered in `registered_parsers`."""

    def __init__(self, cls_name: str):
        super().__init__(f"Class '{cls_name}' has no registered parser to wrap.")


class NotJsonableError(Exception):
    """Raised when tested object is not jsonable."""

    def __init__(self, o: object):
        super().__init__(f"Object {o} is not jsonable!")
