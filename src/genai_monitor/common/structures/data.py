from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, Union

from attrs import define, field
from loguru import logger

from genai_monitor.common.errors import NotJsonableError
from genai_monitor.common.utils import is_jsonable
from genai_monitor.db.schemas.base import BaseModel
from genai_monitor.db.schemas.tables import ArtifactTable, ConditioningTable, ModelTable, SampleTable, UserTable

if TYPE_CHECKING:
    from genai_monitor.query.api import ConditioningQuery, ModelQuery, SampleQuery


@define
class ORMConvertible:
    """A base class to convert between ORM (database model) instances and dataclass instances."""

    _orm_cls: ClassVar[Type[BaseModel]]
    _to_orm_excluded_fields: ClassVar[List[str]]
    _from_orm_excluded_fields: ClassVar[List[str]]
    _relationship_fields: ClassVar[Dict[str, Type["ORMConvertible"]]] = {}

    @classmethod
    def from_orm(cls, orm_instance: BaseModel) -> "ORMConvertible":
        """Gets a dataclass instance from ORM.

        Args:
            orm_instance: The ORM instance corresponding to the dataclass type.

        Returns:
            A dataclass instance.
        """
        field_values = cls._get_field_values_from_orm(orm_instance=orm_instance)
        return cls(**field_values)

    @classmethod
    def _get_field_values_from_orm(cls, orm_instance: BaseModel) -> Dict[str, Any]:
        """Gets values of dataclass fields from its corresponding ORM class.

        Args:
            orm_instance: The instance of ORM class from SQLAlchemy.

        Returns:
            A dictionary mapping the names of fields in the dataclass to their values fetched from ORM instance.

        Raises:
            ValueError if an incompatible ORM class is provided for conversion.
        """
        data_dict = orm_instance.to_dict(nested=True)
        for field_ in cls._relationship_fields:
            field_value = data_dict.get(field_, None)
            if field_value is not None:
                data_dict[field_] = cls._instantiate_related(field_name=field_, value=data_dict[field_])

        field_values = {
            field_: field_value
            for field_, field_value in data_dict.items()
            if (field not in cls._from_orm_excluded_fields) and (not field_.startswith("_"))
        }

        return field_values

    @classmethod
    def _instantiate_related(cls, field_name: str, value: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """Instantiate related dataclasses from dictionaries.

        Args:
            field_name: The name of the field being instantiated.
            value: The dictionary or list of dictionaries to be converted.

        Returns:
            Either a related dataclass instance, a list of instances, or the original value.
        """
        related_cls = cls._relationship_fields.get(field_name)

        if isinstance(related_cls, str):
            related_cls = globals().get(related_cls)

        if isinstance(value, dict) and related_cls:
            return related_cls(**value)
        if isinstance(value, list) and related_cls:
            return [related_cls(**item) for item in value]
        logger.warning(
            f"Could not instantiate a related dataclass. " # type: ignore
            f"Got value {value} for dataclass type {related_cls.__name__}"
        )
        return value

    def to_orm(self) -> BaseModel:
        """Get an ORM object from dataclass.

        Returns:
            And ORM object corresponding to the dataclass.
        """
        field_values = self._get_field_values_from_dataclass()
        orm_instance = self._orm_cls(**field_values)
        self._populate_relationships_to_orm(orm_instance)
        return orm_instance

    def _get_field_values_from_dataclass(self) -> Dict[str, Any]:
        """Get values of ORM class fields from the dataclass.

        Returns:
            A dictionary mapping the names of ORM class fields to their values fetched from dataclass instance.
        """
        return {
            field_name: getattr(self, field_name, None)
            for field_name in type(self)._orm_cls.columns
            if field_name not in type(self)._to_orm_excluded_fields
        }

    def _populate_relationships_to_orm(self, orm_instance: BaseModel):
        """Populates related fields in the ORM instance. Overrides in subclasses to handle specific relationships.

        Args:
            orm_instance: The ORM instance to populate the relationships in
        """



@define
class Conditioning(ORMConvertible):
    """Represents a conditioning in the system and corresponds to ConditioningTable."""

    _orm_cls: ClassVar[Type[ConditioningTable]] = ConditioningTable
    _to_orm_excluded_fields: ClassVar[List[str]] = ["type"]
    _from_orm_excluded_fields: ClassVar[List[str]] = ["type"]
    _relationship_fields: ClassVar[Dict[str, Type["ORMConvertible"]]] = {"samples": "Sample"} # type: ignore

    id: Optional[int] = None
    type_id: Optional[int] = None
    type: Optional[str] = None
    value: Optional[Dict[str, Any]] = field(default=None)
    hash: Optional[str] = None
    value_metadata: Optional[Dict[str, Any]] = field(default=None)
    samples: Optional[List["Sample"]] = None

    @value.validator # type: ignore
    def validate_value(self, attribute, value): # noqa: ANN001,ANN201,ANN401
        if not is_jsonable(value):
            raise NotJsonableError(value)

    def _populate_relationships_to_orm(self, orm_instance: BaseModel):
        if self.samples:
            orm_instance.samples = [sample.to_orm() for sample in self.samples]

    def query(self) -> "ConditioningQuery":
        """Query interface for this conditioning.

        Returns:
            ConditioningQuery: Query interface for this conditioning.
        """
        from genai_monitor.query.api import ConditioningQuery

        return ConditioningQuery(self)


@define
class User(ORMConvertible):
    """Represents a user in the system and corresponds to UsersTable."""

    _orm_cls: ClassVar[Type[UserTable]] = UserTable
    _to_orm_excluded_fields: ClassVar[List[str]] = []
    _from_orm_excluded_fields: ClassVar[List[str]] = []

    id: Optional[int] = None
    name: Optional[str] = None
    hash: Optional[str] = None


@define
class Sample(ORMConvertible):
    """Represents a sample in the system and corresponds to SampleTable."""

    _orm_cls: ClassVar[Type[SampleTable]] = SampleTable
    _to_orm_excluded_fields: ClassVar[List[str]] = []
    _from_orm_excluded_fields: ClassVar[List[str]] = []
    _relationship_fields: ClassVar[Dict[str, Type["ORMConvertible"]]] = {"conditioning": Conditioning}

    id: Optional[int] = None
    model_id: Optional[int] = None
    conditioning_id: Optional[int] = None
    user_id: Optional[int] = None
    name: Optional[str] = None
    hash: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    generation_id: Optional[int] = None
    status: Optional[str] = None
    version: Optional[str] = field(
        default=None, metadata={"description": "GenAI Monitor database version identifier for the sample."}
    )

    data: Optional[bytes] = None
    conditioning: Optional[Conditioning] = None
    artifacts: Optional[List["Artifact"]] = None
    user: Optional[User] = None

    def _populate_relationships_to_orm(self, orm_instance: BaseModel):
        if self.conditioning:
            orm_instance.conditioning = self.conditioning.to_orm()
        if self.user:
            orm_instance.user = self.user.to_orm()
        if self.artifacts:
            orm_instance.artifacts = [artifact.to_orm() for artifact in self.artifacts]

    def query(self) -> "SampleQuery":
        """Query interface for this sample.

        Returns:
            SampleQuery: Query interface for this sample.
        """
        from genai_monitor.query.api import SampleQuery

        return SampleQuery(self)


@define
class Model(ORMConvertible):
    """Represents a generative model and corresponds to GeneratorTable."""

    _orm_cls: ClassVar[Type[ModelTable]] = ModelTable
    _to_orm_excluded_fields: ClassVar[List[str]] = []
    _from_orm_excluded_fields: ClassVar[List[str]] = []

    id: Optional[int] = None
    hash: Optional[str] = None
    model_class: Optional[str] = None
    checkpoint_location: Optional[str] = None
    checkpoint_metadata: Optional[Dict[str, Any]] = None
    training_run_id: Optional[int] = None
    training_step: Optional[int] = None
    model_metadata: Optional[Dict[str, Any]] = None

    def query(self) -> "ModelQuery":
        """Query interface for this generator.

        Returns:
            GeneratorQuery: Query interface for this generator.
        """
        from genai_monitor.query.api import ModelQuery

        return ModelQuery(self)


@define
class Artifact(ORMConvertible):
    """Represents an artifact in the system."""

    _orm_cls: ClassVar[Type[ArtifactTable]] = ArtifactTable
    _to_orm_excluded_fields: ClassVar[List[str]] = []
    _from_orm_excluded_fields: ClassVar[List[str]] = []
    _relationship_fields: ClassVar[Dict[str, Type["ORMConvertible"]]] = {"sample": Sample}

    id: Optional[int] = None
    name: Optional[str] = None
    sample_id: Optional[int] = None
    value: Optional[str] = None
    hash: Optional[str] = None

    sample: Optional[Sample] = None
