import inspect
import json
import textwrap
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import field
from functools import wraps
from typing import Any, Callable, List, Optional, Tuple, Type, Union

from attrs import define
from loguru import logger

from genai_monitor.common.structures.data import Artifact, Conditioning, Model, Sample
from genai_monitor.common.types import SampleStatus
from genai_monitor.db.manager import DBManager
from genai_monitor.db.schemas.tables import (
    ArtifactTable,
    ConditioningTable,
    ConfigurationTable,
    ModelTable,
    SampleEvaluationMapping,
    SampleTable,
)
from genai_monitor.static.constants import EMPTY_MODEL_HASH, UNKNOWN_MODEL_HASH
from genai_monitor.static.fields import CONDITIONING_METADATA_FIELDNAME
from genai_monitor.structures.conditioning_parsers.base import BaseConditioningParser
from genai_monitor.structures.output_parsers.base import BaseModelOutputParser
from genai_monitor.structures.persistency_manager import PersistencyManager
from genai_monitor.structures.runtime_manager import RuntimeManager
from genai_monitor.utils.data_hashing import get_hash_from_jsonable


@define
class Wrapper(ABC):
    """Abstract class for wrapping functions and methods.

    This class is responsible for wrapping functions and methods with listening logic.

    Attributes:
        db_manager: DBManager: The database manager.
        persistency_manager: PersistencyManager: The persistency manager.
        runtime_manager: RuntimeManager: The runtime manager.
        output_parser: BaseModelOutputParser: The output parser.
        conditioning_parser: BaseConditioningParser: The conditioning parser.
        hashing_function: The hashing function.
        func: The wrapped function or method.
    """
    db_manager: DBManager
    persistency_manager: PersistencyManager
    runtime_manager: RuntimeManager
    output_parser: BaseModelOutputParser
    conditioning_parser: BaseConditioningParser
    max_unique_instances: int
    hashing_function: Callable[[Any], str]
    func: Optional[Callable] = field(init=False)

    @abstractmethod
    def wrap(self, func: Callable) -> Callable:
        """Wraps the function or method.

        Args:
            func: The function or method to wrap.

        Returns:
            Callable: The wrapped function or method.
        """

    def _save_sample(
        self,
        model_output: Any,
        conditioning: Conditioning,
        generator: Model,
        generation_id: Optional[Union[int, None]] = None,
        samples_in_conditioning: Optional[List[Sample]] = None,
    ):
        """Saves the sample to the database.

        Args:
            model_output: The output of the model.
            conditioning: The conditioning.
            generator: The generator.
            generation_id: The generation id (from 0 to `max_unique_instances` - 1).
            samples_in_conditioning: The samples in conditioning.
        """
        sample = self.output_parser.get_sample_from_model_output(model_output=model_output)
        sample.model_id = generator.id
        sample.version = self._get_current_version()

        existing_conditioning = self.db_manager.search(ConditioningTable, {"id": conditioning.id})
        if existing_conditioning:
            sample.conditioning_id = conditioning.id
        else:
            sample.conditioning = conditioning

        if self.runtime_manager.user_id:
            sample.user_id = self.runtime_manager.user_id

        if generation_id is not None:
            sample.generation_id = generation_id

        # TODO: check whether data is uploaded to the database
        sample = Sample.from_orm(self.db_manager.save(sample.to_orm()))
        self.runtime_manager.latest_sample = sample
        self.persistency_manager.save_sample(sample)
        logger.info(f"Saved sample #{sample.id} to database.")

        # TODO: Remove this functionality
        if samples_in_conditioning:
            self.db_manager.save(
                instance=SampleEvaluationMapping(
                    evaluated_sample_id=samples_in_conditioning[0].id,
                    evaluation_result_sample_id=sample.id,
                    evaluator_id=generator.id if generator else None,
                )
            )
        self.attach_artifacts_to_sample(sample)

    def _finish_sample_generation(
        self,
        sample: Sample,
        model_output: Any,
        conditioning: Conditioning,
        generator: Model,
        generation_id: Optional[int] = None,
        samples_in_conditioning: Optional[List[Sample]] = None,
    ):
        """Updates sample placeholder with the model output and saves it to the database.

        Args:
            sample: The sample placeholder.
            model_output: The output of the model.
            conditioning: The conditioning.
            generator: The generator.
            generation_id: The generation id (from 0 to `max_unique_instances` - 1).
            samples_in_conditioning: The samples in conditioning.
        """
        updates = {
            "hash": self.output_parser.get_model_output_hash(model_output),
        }

        existing_conditioning = self.db_manager.search(ConditioningTable, {"id": conditioning.id})
        if existing_conditioning:
            updates["conditioning_id"] = conditioning.id
        else:
            updates["conditioning"] = conditioning

        if generator:
            updates["model_id"] = generator.id

        if self.runtime_manager.user_id:
            updates["user_id"] = self.runtime_manager.user_id

        if generation_id is not None:
            updates["generation_id"] = generation_id

        updates["version"] = self._get_current_version()
        updates["status"] = SampleStatus.COMPLETE.value
        sample = Sample.from_orm(
            self.db_manager.update(model=SampleTable, filters={"id": sample.id}, values=updates)[0]
        )
        self.runtime_manager.latest_sample = sample
        sample.data = self.output_parser.model_output_to_bytes(model_output)
        self.persistency_manager.save_sample(sample)
        logger.info(f"Saved sample #{sample.id} to database.")

        if samples_in_conditioning:
            # TODO: only a single evaluated sample is supported for now
            self.db_manager.save(
                instance=SampleEvaluationMapping(
                    evaluated_sample_id=samples_in_conditioning[0].id,
                    evaluation_result_sample_id=sample.id,
                    evaluator_id=generator.id if generator else None,
                )
            )
        self.attach_artifacts_to_sample(sample)

    def _get_generations(self, hash_value: str, conditioning: Conditioning, name: str) -> Tuple[Model, List[Sample]]:
        """Get existing generations from the database.

        Args:
            hash_value: The hash of the model/function.
            conditioning: The conditioning.
            name: The name of the model class/function.

        Returns:
            The generator and the existing generations.
        """
        if hash_value == UNKNOWN_MODEL_HASH:
            logger.error(
                f"Hashing function failed for {name}."
                f"Without a proper hashing function, it will not be possible to find "
                f"corresponding entries in the database."
            )
            generator = None
            generations_complete = self.db_manager.search(
                SampleTable,
                {
                    "conditioning_id": conditioning.id,
                    "model_id": None,
                    "version": self._get_current_version(),
                    "status": SampleStatus.COMPLETE.value,
                },
            )
            generations_in_progress = self.db_manager.search(
                SampleTable,
                {
                    "conditioning_id": conditioning.id,
                    "model_id": None,
                    "version": self._get_current_version(),
                    "status": SampleStatus.IN_PROGRESS.value,
                },
            )
            existing_generations = generations_complete + generations_in_progress
            return generator, existing_generations

        existing_generators = self.db_manager.search(ModelTable, {"hash": hash_value, "model_class": name})
        if not existing_generators:
            logger.info(f"{name} with hash {hash_value} not found in the DB. Registering now.")
            generator = Model(model_class=name, hash=hash_value)
            generator.model_metadata = {"max_unique_instances": self.max_unique_instances}
            generator = Model.from_orm(self.db_manager.save(generator.to_orm()))
        else:
            logger.info(f"Found existing generator with hash {hash_value}.")
            generator = Model.from_orm(existing_generators[0])

        generations_complete = self.db_manager.search(
            SampleTable,
            {
                "model_id": generator.id,
                "conditioning_id": conditioning.id,
                "version": self._get_current_version(),
                "status": SampleStatus.COMPLETE.value,
            },
        )
        generations_in_progress = self.db_manager.search(
            SampleTable,
            {
                "model_id": generator.id,
                "conditioning_id": conditioning.id,
                "version": self._get_current_version(),
                "status": SampleStatus.IN_PROGRESS.value,
            },
        )
        existing_generations = generations_complete + generations_in_progress
        return generator, existing_generations

    def _return_existing_generation(
        self,
        existing_generations: List[Sample],
        samples_in_conditioning: List[Sample],
        generation_id: Optional[int] = None,
        generator: Model = None,
    ) -> Any:
        """Handle existing generations.

        Args:
            existing_generations: The existing generations.
            samples_in_conditioning: The samples in conditioning.
            hash_value: The hash of the model/function.
            generation_id: The generation id.
            generator: The generator.

        Returns:
            Any: The output of the model.

        Raises:
            FileNotFoundError: If the data cannot be loaded from disk.
        """
        logger.info("Found existing generations, loading data from disk.")

        samples = [sample for sample in existing_generations if Sample.from_orm(sample).generation_id == generation_id]
        if not samples:
            logger.error(f"Could not find generation with id {generation_id}")
            return None
        if len(samples) > 1:
            logger.error(f"Found multiple generations with id {generation_id}")
            return None

        sample = samples[0]
        while sample.status != SampleStatus.COMPLETE.value:
            logger.info(f"Generation {sample.id} is not complete. Waiting for completion...")
            time.sleep(1)
            sample = self.db_manager.search(SampleTable, {"id": sample.id})[0]

        if samples_in_conditioning:
            self.db_manager.save(
                instance=SampleEvaluationMapping(
                    evaluated_sample_id=samples_in_conditioning[0].id,
                    evaluation_result_sample_id=sample.id,
                    evaluator_id=generator.id if generator else None,
                )
            )

        try:
            self.runtime_manager.latest_sample = sample
            sample.data = self.persistency_manager.load_sample(sample)
            model_output = self.output_parser.get_model_output_from_sample(sample)
            self.attach_artifacts_to_sample(sample)
            return model_output

        except FileNotFoundError as e:
            logger.error(f"Failed to load data from disk: {e}")

    def _resolve_conditioning(self, conditioning: Conditioning) -> Conditioning:
        existing_conditioning = self.db_manager.search(ConditioningTable, {"hash": conditioning.hash})

        if existing_conditioning:
            conditioning_text = textwrap.shorten(json.dumps(conditioning.value), width=200, placeholder="...")
            logger.info(f"Found existing conditioning with value {conditioning_text}.")
            conditioning = Conditioning.from_orm(existing_conditioning[0])

            if self.persistency_manager.enabled:
                conditioning.value = self.persistency_manager.load_conditioning(conditioning)
        else: # noqa: PLR5501
            if self.persistency_manager.enabled:
                value = conditioning.value
                conditioning.value = None
                conditioning = Conditioning.from_orm(self.db_manager.save(conditioning.to_orm()))
                conditioning.value = value
                self.persistency_manager.save_conditioning(conditioning)
            else:
                conditioning = Conditioning.from_orm(self.db_manager.save(conditioning.to_orm()))

        return conditioning

    def attach_artifacts_to_sample(self, sample: Sample) -> None:
        """Attach artifacts to a sample.

        Args:
            sample: The sample to attach artifacts to.
        """
        while self.runtime_manager.artifacts_for_next_sample:
            artifact = self.runtime_manager.artifacts_for_next_sample.popleft()

            existing_artifacts = self.db_manager.search(
                ArtifactTable, {"name": artifact.name, "hash": artifact.hash, "sample_id": sample.id}
            )

            if existing_artifacts:
                logger.info(f"Artifact {artifact.name} already exists in the database.")

            else: # noqa: PLR5501
                if self.persistency_manager.enabled:
                    value = artifact.value
                    artifact.value = None
                    artifact.sample_id = sample.id
                    artifact.hash = get_hash_from_jsonable(value)
                    artifact = Artifact.from_orm(self.db_manager.save(artifact.to_orm()))
                    artifact.value = value
                    self.persistency_manager.save_artifact(artifact)
                    logger.info(f"Saved artifact #{artifact.id} (sample #{sample.id}) to database and disk.")
                else:
                    artifact = Artifact.from_orm(self.db_manager.save(artifact.to_orm()))
                    logger.info(f"Saved artifact #{artifact.id} (sample #{sample.id}) to database.")

    def update_generation_id(self, conditioning: Conditioning, generation_id: int):
        """Update the generation_id of the most recently used generation.

        Args:
            conditioning: The conditioning.
            generation_id: The generation id.
        """
        conditioning_metadata = conditioning.value_metadata
        conditioning_metadata["latest_instance"] = generation_id

        self.db_manager.update(
            model=ConditioningTable, filters={"id": conditioning.id}, values={"value_metadata": conditioning_metadata}
        )

    def _get_current_version(self) -> str:
        if self.runtime_manager.version is not None:
            return self.runtime_manager.version

        version = self.db_manager.search(ConfigurationTable, {"key": "version"})

        if not version:
            raise ValueError("Database version not found.")

        if len(version) > 1:
            raise ValueError("Multiple database versions found.")

        return version[0].to_dict().get("value")

    def _create_sample_placeholder(
        self, conditioning: Conditioning, generator: Model, generation_id: Optional[int] = None
    ) -> Sample:
        """Create a placeholder for a sample.

        Args:
            conditioning: The conditioning.
            generator: The generator.
            generation_id: The generation id.

        Returns:
            The placeholder sample with status: "In progress".
        """
        sample = Sample(
            conditioning_id=conditioning.id,
            model_id=generator.id,
            generation_id=generation_id,
            status=SampleStatus.IN_PROGRESS.value,
            hash=EMPTY_MODEL_HASH,
            version=self._get_current_version(),
        )
        sample = Sample.from_orm(self.db_manager.save(sample.to_orm()))
        return sample


@define
class FunctionWrapper(Wrapper):
    """Wrapper for functions."""

    def wrap(self, func: Callable) -> Callable:
        """Wraps the function.

        Args:
            func: The function to wrap.

        Returns:
            Callable: The wrapped function.
        """
        self.func = deepcopy(func)

        @wraps(func)
        def wrapped(*args, **kwargs) -> Any:
            function_hash = self.hashing_function(func)
            function_name = func.__name__

            conditioning, samples_in_conditioning = self.conditioning_parser.parse_conditioning(func, *args, **kwargs)
            kwargs.pop(CONDITIONING_METADATA_FIELDNAME, None)

            conditioning = self._resolve_conditioning(conditioning)
            generator, existing_generations = self._get_generations(function_hash, conditioning, function_name)

            max_unique_instances = generator.model_metadata.get("max_unique_instances", 1)
            next_instance = (conditioning.value_metadata.get("latest_instance", -1) + 1) % max_unique_instances

            if not self.persistency_manager.enabled:
                logger.info("PersistencyManager is disabled. Output will be generated.")
                model_output = func(*args, **kwargs)

                if not existing_generations:
                    sample = self._create_sample_placeholder(conditioning=conditioning, generator=generator)
                    self._finish_sample_generation(
                        sample=sample,
                        model_output=model_output,
                        conditioning=conditioning,
                        generator=generator,
                        samples_in_conditioning=samples_in_conditioning,
                    )

                return model_output

            if len(existing_generations) >= max_unique_instances:
                logger.info(f"Max instances ({max_unique_instances}) reached. Returning existing generation.")
                try:
                    model_output = self._return_existing_generation(
                        existing_generations=existing_generations,
                        samples_in_conditioning=samples_in_conditioning,
                        generator=generator,
                        generation_id=next_instance,
                    )

                    self.update_generation_id(conditioning, next_instance)

                    return model_output

                except Exception as e:
                    logger.error(f"Could not return existing generation: {e}")
                    logger.info("Generating new sample without sample creation.")
                    return func(*args, **kwargs)

            sample_placeholder = self._create_sample_placeholder(
                conditioning=conditioning, generator=generator, generation_id=next_instance
            )

            try:
                model_output = func(*args, **kwargs)

            except Exception as e:
                self.db_manager.update(
                    model=SampleTable,
                    filters={"id": sample_placeholder.id},
                    values={"status": SampleStatus.FAILED.value},
                )
                logger.error(f"Could not generate sample: {e}")
                raise e

            self._finish_sample_generation(
                sample=sample_placeholder,
                model_output=model_output,
                conditioning=conditioning,
                generator=generator,
                samples_in_conditioning=samples_in_conditioning,
                generation_id=next_instance,
            )
            self.update_generation_id(conditioning, next_instance)
            return model_output

        return wrapped


@define
class MethodWrapper(Wrapper):
    """Wrapper for class methods."""

    def wrap(self, func: Callable) -> Callable:
        self.func = deepcopy(func)

        @wraps(func)
        def wrapped(obj_self, *args, **kwargs) -> Any:
            model_hash = self.hashing_function(obj_self)
            model_cls_name = obj_self.__class__.__name__
            conditioning, samples_in_conditioning = self.conditioning_parser.parse_conditioning(func, *args, **kwargs)
            kwargs.pop(CONDITIONING_METADATA_FIELDNAME, None)

            conditioning = self._resolve_conditioning(conditioning)
            generator, existing_generations = self._get_generations(model_hash, conditioning, model_cls_name)

            max_unique_instances = generator.model_metadata.get("max_unique_instances", 1)
            next_instance = (conditioning.value_metadata.get("latest_instance", -1) + 1) % max_unique_instances

            if not self.persistency_manager.enabled:
                logger.info("PersistencyManager is disabled. Output will be generated.")
                model_output = func(obj_self, *args, **kwargs)

                if not existing_generations:
                    sample = self._create_sample_placeholder(conditioning=conditioning, generator=generator)
                    self._finish_sample_generation(
                        sample=sample,
                        model_output=model_output,
                        conditioning=conditioning,
                        generator=generator,
                        samples_in_conditioning=samples_in_conditioning,
                    )

                return model_output

            if len(existing_generations) >= max_unique_instances:
                logger.info(f"Max instances ({max_unique_instances}) reached. Returning existing generation")

                model_output = self._return_existing_generation(
                    existing_generations=existing_generations,
                    samples_in_conditioning=samples_in_conditioning,
                    generator=generator,
                    generation_id=next_instance,
                )

                self.update_generation_id(conditioning, next_instance)

                return model_output

            sample_placeholder = self._create_sample_placeholder(
                conditioning=conditioning, generator=generator, generation_id=next_instance
            )

            try:
                model_output = func(obj_self, *args, **kwargs)

            except Exception as e:
                self.db_manager.update(
                    SampleTable, {"id": sample_placeholder.id}, {"status": SampleStatus.FAILED.value}
                )
                logger.error(f"Could not generate sample: {e}")

            self._finish_sample_generation(
                sample=sample_placeholder,
                model_output=model_output,
                conditioning=conditioning,
                generator=generator,
                samples_in_conditioning=samples_in_conditioning,
                generation_id=next_instance,
            )
            self.update_generation_id(conditioning, next_instance)
            return model_output

        return wrapped


class WrapperFactory:
    """Factory for creating wrappers for functions and methods."""

    @staticmethod
    def create(
        func: Callable,
        db_manager: DBManager,
        persistency_manager: PersistencyManager,
        runtime_manager: RuntimeManager,
        output_parser: BaseModelOutputParser,
        conditioning_parser: BaseConditioningParser,
        hashing_function: Optional[Callable[[object], str]] = None,
        max_unique_instances: int = 1,
    ) -> Union[FunctionWrapper, MethodWrapper]:
        """Creates a wrapper for a function or method.

        Args:
            func: The function or method to wrap.
            db_manager: The database manager.
            persistency_manager: The persistency manager.
            runtime_manager: The runtime manager.
            output_parser: The output parser.
            conditioning_parser: The conditioning parser.
            hashing_function: The hashing function.
            max_unique_instances: The maximum number of unique sample instances for each conditioning.

        Returns:
            The wrapper for the function or method.
        """
        if is_class_func(func=func):
            return MethodWrapper(
                db_manager=db_manager,
                persistency_manager=persistency_manager,
                runtime_manager=runtime_manager,
                output_parser=output_parser,
                conditioning_parser=conditioning_parser,
                hashing_function=hashing_function,
                max_unique_instances=max_unique_instances,
            )

        return FunctionWrapper(
            db_manager=db_manager,
            persistency_manager=persistency_manager,
            runtime_manager=runtime_manager,
            output_parser=output_parser,
            conditioning_parser=conditioning_parser,
            hashing_function=hashing_function,
            max_unique_instances=max_unique_instances,
        )


def is_class_func(func: Callable) -> bool:
    """Check if a function is a class method.

    Args:
        func: The function to check.

    Returns:
        bool: True if the function is a class method, False otherwise.
    """
    return get_defining_class(func=func) is not None


def get_defining_class(func: Callable) -> Optional[Type]:
    """Get the class that defines a method.

    Args:
        func: The function to check.

    Returns:
        The class that defines the method.
    """
    parts = func.__qualname__.split(".")
    if len(parts) > 1:
        class_name = parts[-2]
        module = inspect.getmodule(func)
        if module and hasattr(module, class_name):
            return getattr(module, class_name)
    return None


@define
class ArtifactWrapper(ABC):
    """Abstract class for wrapping functions and methods for artifact tracking.

    Attributes:
        db_manager: The database manager.
        runtime_manager: The runtime manager.
    """
    db_manager: DBManager
    persistency_manager: PersistencyManager
    runtime_manager: RuntimeManager

    def _add_artifact_to_queue(self, output: Any, name: str):
        """Adds an artifact to the runtime manager's queue.

        Args:
            output: The output to add as an artifact.
            name: The name of the artifact.
        """
        artifact = Artifact(
            name=name,
            value=json.dumps(output),
            hash=get_hash_from_jsonable(output),
        )
        self.runtime_manager.artifacts_for_next_sample.append(artifact)

    def _save_artifact(self, output: Any, name: str):
        """Saves an artifact to the database.

        Args:
            output: The output to save as an artifact.
            name: The name of the artifact.

        """
        if not self.runtime_manager.latest_sample:
            logger.error("Cannot attach artifact to a sample. No sample found.")
            return

        sample_id = self.runtime_manager.latest_sample.id
        if not sample_id:
            logger.error("Cannot attach artifact to a sample. No sample id found for the latest sample.")
            return
        value = json.dumps(output)
        hashvalue = get_hash_from_jsonable(output)
        existing_artifacts = self.db_manager.search(
            ArtifactTable, {"name": name, "hash": hashvalue, "sample_id": sample_id}
        )

        if existing_artifacts:
            logger.info(f"Artifact {name} already exists in the database.")
            return

        artifact = Artifact(
            name=name,
            value=value,
            hash=hashvalue,
            sample_id=sample_id,
        )

        if self.persistency_manager.enabled:
            artifact.value = None
            artifact = Artifact.from_orm(self.db_manager.save(artifact.to_orm()))
            artifact.value = value
            self.persistency_manager.save_artifact(artifact)
            logger.info(f"Saved artifact #{artifact.id} (sample #{sample_id}) to database and disk.")
        else:
            artifact = Artifact.from_orm(self.db_manager.save(artifact.to_orm()))
            logger.info(f"Saved artifact #{artifact.id} (sample #{sample_id}) to database.")


class ArtifactFunctionWrapper(ArtifactWrapper):
    """Wrapper for functions for artifact tracking."""

    def wrap_forward(self, func: Callable) -> Callable:
        """Wraps the function for forward artifact tracking.

        Args:
            func: The function to wrap.

        Returns:
            The wrapped function.
        """
        self.func = deepcopy(func)

        @wraps(func)
        def wrapped(*args, **kwargs) -> Any:
            output = func(*args, **kwargs)
            self._add_artifact_to_queue(output=output, name=f"{func.__module__}.{func.__name__}")
            return output

        return wrapped

    def wrap_backward(self, func: Callable) -> Callable:
        """Wraps the function for backward artifact tracking.

        Args:
            func: The function to wrap.

        Returns:
            The wrapped function.
        """
        self.func = deepcopy(func)

        @wraps(func)
        def wrapped(*args, **kwargs) -> Any:
            output = func(*args, **kwargs)
            self._save_artifact(output, name=f"{func.__module__}.{func.__name__}")
            return output

        return wrapped


class ArtifactMethodWrapper(ArtifactWrapper):
    """Wrapper for methods for artifact tracking."""

    def wrap_forward(self, func: Callable) -> Callable:
        """Wraps the function for forward artifact tracking.

        Args:
            func: The function to wrap.

        Returns:
            The wrapped function.
        """
        self.func = deepcopy(func)

        @wraps(func)
        def wrapped(obj_self, *args, **kwargs) -> Any:
            output = func(*args, **kwargs)
            self._add_artifact_to_queue(output=output, name=f"{obj_self.__class__.__name__}.{func.__name__}")
            return output

        return wrapped

    def wrap_backward(self, func: Callable) -> Callable:
        """Wraps the function for backward artifact tracking.

        Args:
            func: The function to wrap.

        Returns:
            The wrapped function.
        """
        self.func = deepcopy(func)

        @wraps(func)
        def wrapped(obj_self, *args, **kwargs) -> Any:
            output = func(*args, **kwargs)
            self._save_artifact(output, name=f"{obj_self.__class__.__name__}.{func.__name__}")
            return output

        return wrapped


class ArtifactWrapperFactory:
    """Factory for creating wrappers for artifacts."""
    def create(
        self,
        func: Callable,
        db_manager: DBManager,
        persistency_manager: PersistencyManager,
        runtime_manager: RuntimeManager,
    ) -> Union[ArtifactFunctionWrapper, ArtifactMethodWrapper]:
        """Creates a wrapper for an artifact.

        Args:
            func: The artifact to wrap.
            db_manager: The database manager.
            persistency_manager: The persistency manager.
            runtime_manager: The runtime manager.

        Returns:
            The wrapper for the artifact.
        """
        if is_class_func(func=func):
            return ArtifactMethodWrapper(
                db_manager=db_manager, persistency_manager=persistency_manager, runtime_manager=runtime_manager
            )

        return ArtifactFunctionWrapper(
            db_manager=db_manager, persistency_manager=persistency_manager, runtime_manager=runtime_manager
        )
