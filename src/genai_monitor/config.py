from pydantic_settings import BaseSettings


class PersistencyManagerConfig(BaseSettings):
    """The configuration for the persistency manager."""

    enabled: bool
    path: str


class DBManagerConfig(BaseSettings):
    """The configuration for the database manager."""

    url: str


class Config(BaseSettings):
    """The configuration for the hidden mode."""

    persistency: PersistencyManagerConfig
    db: DBManagerConfig
    version: str
