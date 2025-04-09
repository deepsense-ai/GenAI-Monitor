import pytest
import torch

from genai_monitor.common.errors import NotJsonableError
from genai_monitor.common.structures.data import Conditioning


@pytest.fixture
def not_jsonable_dict_with_tensor():
    return {"input_tensor": torch.Tensor([1, 2, 3])}


def test_conditioning_creation_from_not_jsonable(not_jsonable_dict_with_tensor):
    with pytest.raises(NotJsonableError, match=r"Object {'input_tensor': tensor\(\[1\., 2\., 3\.]\)} is not jsonable!"):
        Conditioning(value=not_jsonable_dict_with_tensor)


def test_conditioning_creation_from_not_jsonable_using_parser(
    conditioning_parser_for_tensors, not_jsonable_dict_with_tensor
):
    parsed_arguments = conditioning_parser_for_tensors.parse_func_arguments(**not_jsonable_dict_with_tensor)
    Conditioning(value=parsed_arguments)
