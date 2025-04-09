from genai_monitor.utils.model_hashing import default_model_hashing_function
from tests.conftest import DummyPytorchModel


def test_pytorch_model_hashing():
    model = DummyPytorchModel()
    model_hash = default_model_hashing_function(model)
    assert isinstance(model_hash, str)
