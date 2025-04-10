from sqlalchemy.orm import DeclarativeBase, declarative_base
from sqlalchemy_mixins.repr import ReprMixin
from sqlalchemy_mixins.serialize import SerializeMixin

Base: DeclarativeBase = declarative_base()


class BaseModel(Base, SerializeMixin, ReprMixin):  # type: ignore
    """Base model for all database tables."""

    __abstract__ = True
