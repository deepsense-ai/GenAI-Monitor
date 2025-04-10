from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from .schemas.base import BaseModel
from .schemas.tables import (  # noqa: F401
    ConditioningTable,
    ConditioningTypeTable,
    ModelTable,
    SampleEvaluationMapping,
    SampleTable,
)

DEFAULT_DATABASE_URL = "sqlite:///genai_eval.db"


class SessionManager:
    """Manages the database engine and provides session management."""

    _engine: Optional[Engine] = None
    _session_factory: Optional[sessionmaker] = None

    def __init__(self, database_url: str = DEFAULT_DATABASE_URL):  # noqa: ANN204,D107
        self.initialize(database_url=database_url)

    def initialize(self, database_url: str = DEFAULT_DATABASE_URL) -> "SessionManager":
        """Initializes the database engine and session factory. This should be called once on application start.

        Args:
            database_url: The URL for the database connection. Defaults to sqlite:///genai_eval.db.

        Returns:
            The `SessionManager` object.
        """
        if self._engine is None:
            self._engine = create_engine(database_url)
            self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
            BaseModel.metadata.create_all(bind=self._engine)

        return self

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations. Commits or rolls back on error.

        Raises:
            Exception: on any error during database transaction
            ConnectionError: if the database connection has not been initialized yet

        Yields:
            The database connection session
        """
        if self._session_factory is None:
            raise ConnectionError(
                "The connection to the database must be initialized through, SessionManager.initialize()"
            )

        session = self._session_factory(expire_on_commit=False)
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
