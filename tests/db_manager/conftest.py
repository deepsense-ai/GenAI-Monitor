import pytest

from genai_monitor.common.types import SampleStatus
from genai_monitor.db.config import SessionManager
from genai_monitor.db.manager import DBManager
from genai_monitor.db.schemas.tables import ConditioningTable, ConditioningTypeTable, ModelTable, SampleTable

TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(autouse=True)
def session_manager():
    """Provide a SessionManager instance for use in tests."""
    return SessionManager(database_url=TEST_DATABASE_URL)


@pytest.fixture
def db_manager(session_manager):
    """Provide a DBManager instance for use in tests."""
    return DBManager(session_manager=session_manager)


@pytest.fixture
def db_session(session_manager):
    """Provide a SQLAlchemy session for use in tests, with rollback and close at the end."""
    with session_manager.session_scope() as session:
        yield session
        session.rollback()


@pytest.fixture
def setup_database_with_data(db_session):
    """Database setup.

    Populate the test database with initial data, providing tables and records for tests to use.
    Cleanup occurs after all tests have completed.
    """
    conditioning_type1 = ConditioningTypeTable(type="Type A")
    conditioning_type2 = ConditioningTypeTable(type="Type B")
    db_session.add_all([conditioning_type1, conditioning_type2])
    db_session.commit()

    conditioning1 = ConditioningTable(type_id=conditioning_type1.id, value={"key": "value1"}, hash="value1")
    conditioning2 = ConditioningTable(type_id=conditioning_type2.id, value={"key": "value2"}, hash="value1")
    db_session.add_all([conditioning1, conditioning2])
    db_session.commit()

    generator = ModelTable(
        hash="gen_hash",
        model_class="ModelClass",
        checkpoint_location="path/to/checkpoint",
    )

    db_session.add(generator)
    db_session.commit()

    sample1 = SampleTable(
        conditioning_id=conditioning1.id,
        name="Sample 1",
        hash="abc123",
        meta={"meta_key": "meta_value1"},
        model_id=generator.id,
        version="test",
        status=SampleStatus.COMPLETE.value,
    )
    sample2 = SampleTable(
        conditioning_id=conditioning2.id,
        name="Sample 2",
        hash="def456",
        meta={"meta_key": "meta_value2"},
        model_id=generator.id,
        version="test",
        status=SampleStatus.COMPLETE.value,
    )
    db_session.add_all([sample1, sample2])
    db_session.commit()

    yield

    db_session.query(SampleTable).delete()
    db_session.query(ConditioningTable).delete()
    db_session.query(ConditioningTypeTable).delete()
    db_session.query(ModelTable).delete()
    db_session.commit()
