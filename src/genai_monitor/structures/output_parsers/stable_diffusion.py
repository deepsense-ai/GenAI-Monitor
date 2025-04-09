from genai_monitor.dependencies import DIFFUSERS_AVAILABLE, EXTRAS_REQUIRE, require_extra

if DIFFUSERS_AVAILABLE:
    import io
    from typing import List, NewType, Tuple, Union

    from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
    from PIL import Image

    from genai_monitor.structures.output_parsers.base import BaseModelOutputParser

    StableDiffusionOutputType = NewType(
        "StableDiffusionOutputType", Union[StableDiffusionPipelineOutput, Tuple[List[Image.Image], List[bool]]]
    )

    class StableDiffusionOutputParser(BaseModelOutputParser[StableDiffusionOutputType]):
        """Output parser for the Stable Diffusion class models."""

        def __init__(self): # noqa: D107,ANN204
            require_extra("diffusers", EXTRAS_REQUIRE)
            super().__init__()

        def model_output_to_bytes(self, model_output: StableDiffusionOutputType) -> bytes:  # type: ignore
            """Converts the model output to a byte representation.

            Args:
                model_output: The model output to convert.

            Returns:
                The byte representation of the model output.
            """
            image = self._get_image_from_model_output(model_output)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()

        def bytes_to_model_output(self, databytes: bytes) -> StableDiffusionOutputType:  # type: ignore
            """Converts a byte representation back into model output.

            Args:
                databytes: The byte representation of the model output.

            Returns:
                The model output reconstructed from the byte representation.
            """
            image = Image.open(io.BytesIO(databytes))
            return StableDiffusionPipelineOutput(images=[image], nsfw_content_detected=[])

        def _get_image_from_model_output(self, model_output: StableDiffusionOutputType) -> Image.Image:  # type: ignore
            """Extracts a single image from the model output.

            Args:
                model_output: The model output to extract the image from.

            Returns:
                The single extracted image.

            Raises:
                ValueError: If the model output is of an unsupported type.
            """
            if isinstance(model_output, Tuple):
                return self._get_data_from_tuple(model_output)

            if isinstance(model_output, StableDiffusionPipelineOutput):
                return self._get_data_from_pipeline_output(model_output)

            raise ValueError(
                f"Model output can be either Tuple or StableDiffusionPipelineOutput, but got {type(model_output)}."
            )

        def _get_data_from_tuple(self, model_output: Tuple[List[Image.Image], List[bool]]) -> Image.Image:
            self._contains_single_image(model_output[0])
            return model_output[0][0]

        def _get_data_from_pipeline_output(self, model_output: StableDiffusionPipelineOutput) -> Image.Image:
            self._contains_single_image(model_output.images)
            return model_output.images[0]

        @staticmethod
        def _contains_single_image(imgs: List[Image.Image]) -> None:
            """Ensures that the model output contains a single image.

            Args:
                imgs: A list of images.

            Raises:
                ValueError: If more than one image is present.
            """
            if len(imgs) > 1:
                raise ValueError(
                    "Only single output generations are supported. "
                    "For multiple images, rerun the generations individually."
                )
