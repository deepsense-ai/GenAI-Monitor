from typing import List, Optional

from sqlalchemy import JSON, ForeignKey, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class ConditioningTypeTable(BaseModel):
    """
    Database table representing the type of conditioning.
    """

    __tablename__ = "conditioning_type"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column()


class ConditioningTable(BaseModel):
    """
    Database table representing the value of conditioning.
    """

    __tablename__ = "conditioning"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # TODO: mark as not optional when conditioning types are registered with conditionings
    type_id: Mapped[Optional[str]] = mapped_column(ForeignKey("conditioning_type.id"))
    value: Mapped[dict] = mapped_column(JSON)
    hash: Mapped[str] = mapped_column()
    value_metadata: Mapped[Optional[dict]] = mapped_column(JSON)

    samples: Mapped[List["SampleTable"]] = relationship(back_populates="conditioning")


class SampleTable(BaseModel):
    """
    Database table representing the sample - an atomic unit of data in the system.
    """

    __tablename__ = "sample"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[Optional[int]] = mapped_column(ForeignKey("model.id"))
    conditioning_id: Mapped[Optional[int]] = mapped_column(ForeignKey("conditioning.id"))
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("user.id"))
    name: Mapped[Optional[str]] = mapped_column()
    hash: Mapped[str] = mapped_column()
    meta: Mapped[Optional[dict]] = mapped_column(JSON)
    generation_id: Mapped[Optional[int]] = mapped_column()
    status: Mapped[str] = mapped_column()
    version: Mapped[str] = mapped_column()

    conditioning: Mapped["ConditioningTable"] = relationship(back_populates="samples")
    user: Mapped["UserTable"] = relationship(back_populates="samples")
    artifacts: Mapped[List["ArtifactTable"]] = relationship(back_populates="sample")


class ModelTable(BaseModel):
    """
    Database table representing the generative model.
    """

    __tablename__ = "model"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    hash: Mapped[Optional[str]] = mapped_column()
    model_class: Mapped[Optional[str]] = mapped_column()
    checkpoint_location: Mapped[Optional[str]] = mapped_column()
    checkpoint_metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    training_step: Mapped[Optional[int]] = mapped_column()
    model_metadata: Mapped[Optional[dict]] = mapped_column(JSON)


class SampleEvaluationMapping(BaseModel):
    """
    Database table representing association between an evaluated sample and its evaluation result.
    """

    __tablename__ = "sample_evaluation_mapping"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    evaluated_sample_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sample.id"))
    evaluation_result_sample_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sample.id"))
    evaluator_id: Mapped[Optional[int]] = mapped_column(ForeignKey("model.id"))


class ConfigurationTable(BaseModel):
    """
    Database table representing system configuration.
    Default values are set at the database level to ensure consistency across all instances.
    """

    __tablename__ = "configuration"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(unique=True)  # e.g. "persistency.path", "db.path"
    value: Mapped[str] = mapped_column()  # string representation of the value
    description: Mapped[Optional[str]] = mapped_column()
    updated_at: Mapped[str] = mapped_column()  # ISO format timestamp
    is_default: Mapped[bool] = mapped_column(server_default=text("0"))  # whether this is a default value
    default_value: Mapped[Optional[str]] = mapped_column()  # default value if any

    __table_args__ = (UniqueConstraint("key", name="uq_configuration_key"),)


class UserTable(BaseModel):
    """
    Database table representing the users.
    """

    __tablename__ = "user"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column()
    hash: Mapped[str] = mapped_column()

    samples: Mapped[List["SampleTable"]] = relationship(back_populates="user")


class ArtifactTable(BaseModel):
    """
    Database table representing the artifact - an atomic unit of data in the system.
    """

    __tablename__ = "artifact"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column()
    value: Mapped[Optional[str]] = mapped_column()
    hash: Mapped[str] = mapped_column()

    sample_id: Mapped[int] = mapped_column(ForeignKey("sample.id"))
    sample: Mapped["SampleTable"] = relationship(back_populates="artifacts")
