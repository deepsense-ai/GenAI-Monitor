from loguru import logger

from genai_monitor.utils.model_hashing import default_model_hashing_function, get_component_hash
from tests.conftest import DummyPytorchModel


def test_pytorch_model_hashing():
    model = DummyPytorchModel()
    model_hash = default_model_hashing_function(model)
    assert isinstance(model_hash, str)


def test_dummy_component_hashing_pytest():
    """Dummy component hashing test using pytest.

    This test verifies that changing an attribute on a dummy class
    results in a different hash when using get_component_hash.
    """
    class DummyComponent:
        def __init__(self, field1, field2):
            self.field1 = field1
            self.field2 = field2

    comp = DummyComponent(field1="initial", field2=123)

    initial_hash = get_component_hash(comp)
    assert isinstance(initial_hash, str), "Initial hash should be a string"
    assert len(initial_hash) > 0, "Initial hash should not be empty"
    comp.field2 = 999
    changed_hash = get_component_hash(comp)

    assert initial_hash != changed_hash, "Changing a field should result in a different hash"

    logger.info(f"Initial Hash: {initial_hash}, Changed Hash: {changed_hash}")
