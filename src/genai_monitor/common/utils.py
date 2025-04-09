def is_jsonable(o: object) -> bool:
    """Check if object can be serialized to json.

    Args:
        o: Object to serialize.

    Returns:
        True if the object can be serialized, false otherwise.
    """
    try:
        import json

        json.dumps(o)
        return True
    except (TypeError, OverflowError):
        return False
