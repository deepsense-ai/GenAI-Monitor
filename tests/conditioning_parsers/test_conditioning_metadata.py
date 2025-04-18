import torch

from genai_monitor.common.structures.data import Conditioning
from genai_monitor.db.schemas.tables import ConditioningTable, SampleTable
from genai_monitor.registration.api import register_class
from genai_monitor.static.fields import CONDITIONING_METADATA_FIELDNAME
from genai_monitor.utils.auto_mode_configuration import load_config
from tests.conftest import DummyPytorchModel


def test_conditioning_metadata_caching(container, tmp_settings, get_pytorch_model_params_for_registration):
    config = load_config(tmp_settings.get("db.url")["value"], tmp_settings)
    container.config.from_dict(config.model_dump())
    container.wire(["genai_monitor.registration.api"])
    registration_params = get_pytorch_model_params_for_registration
    registration_params["cls"] = DummyPytorchModel
    db_manager = container.db_manager()
    register_class(**registration_params)
    x = torch.randn(1, 10)
    model_instance = DummyPytorchModel()
    n_samples = len(db_manager.search(SampleTable))
    n_conditionings = len(db_manager.search(ConditioningTable))
    # Different metadata should yield separate samples
    _ = model_instance.forward(**{"x": x, CONDITIONING_METADATA_FIELDNAME: {"test_id": 1}})
    _ = model_instance.forward(**{"x": x, CONDITIONING_METADATA_FIELDNAME: {"test_id": 2}})

    assert len(db_manager.search(SampleTable)) == n_samples + 2
    assert len(db_manager.search(ConditioningTable)) == n_conditionings + 2

    # Retrieval can be done by the metadata
    _ = model_instance.forward(**{"x": x, CONDITIONING_METADATA_FIELDNAME: {"test_id": 1}})
    _ = model_instance.forward(**{"x": x, CONDITIONING_METADATA_FIELDNAME: {"test_id": 2}})

    assert len(db_manager.search(SampleTable)) == n_samples + 2
    assert len(db_manager.search(ConditioningTable)) == n_conditionings + 2

    conditionings = db_manager.search(ConditioningTable)
    conditionings = [Conditioning.from_orm(conditioning) for conditioning in conditionings]

    persistency_manager = container.persistency_manager()
    for conditioning in conditionings:
        conditioning.value = persistency_manager.load_conditioning(conditioning)
        print(conditioning.value)

    conditionings = [
        cond for cond in conditionings if cond.value["genai_monitor_metadata"].get("test_id", None) is not None
    ]
    assert len(conditionings) == 2
