import pytest

from genai_monitor.db.schemas.tables import ConditioningTable, ConditioningTypeTable, ModelTable, SampleTable


def test_save(db_manager, setup_database_with_data):
    new_conditioning_type = ConditioningTypeTable(type="New Type")
    result = db_manager.save(new_conditioning_type)
    assert result.id is not None


def test_search_with_filters(db_manager, setup_database_with_data):
    filters = {"name": "Sample 1"}
    result = db_manager.search(SampleTable, filters)
    assert len(result) == 1
    assert result[0].name == "Sample 1"


def test_search_without_filters(db_manager, setup_database_with_data):
    result = db_manager.search(SampleTable)
    assert len(result) == 2


def test_update_with_instance(db_manager, db_session, setup_database_with_data):
    instance = db_session.query(SampleTable).filter_by(name="Sample 1").first()
    db_session.expunge(instance)
    values = {"name": "Updated Sample"}
    updated_instance = db_manager.update(instance=instance, values=values)
    assert updated_instance.name == "Updated Sample"


def test_update_with_model_and_filters(db_manager, db_session, setup_database_with_data):
    filters = {"name": "Sample 2"}
    values = {"name": "Updated Sample 2"}
    rows_updated = db_manager.update(model=SampleTable, filters=filters, values=values)
    assert len(rows_updated) == 1
    updated_instance = db_session.query(SampleTable).filter_by(name="Updated Sample 2").first()
    assert updated_instance is not None


def test_update_raises_value_error_without_instance_or_model(db_manager):
    with pytest.raises(ValueError):
        db_manager.update(filters={"name": "Sample 1"}, values={"name": "Updated Sample"})


def test_join_search_with_filters(db_manager, setup_database_with_data):
    on_condition = SampleTable.model_id == ModelTable.id
    target_filters = {"conditioning_id": 1}

    results = db_manager.join_search(
        target_model=SampleTable,
        join_model=ModelTable,
        on_condition=on_condition,
        target_filters=target_filters,
    )

    assert len(results) > 0
    for result in results:
        assert result.conditioning_id == 1
        assert result.model_id is not None


def test_join_search_without_filters(db_manager, setup_database_with_data):
    on_condition = SampleTable.conditioning_id == ConditioningTable.id
    results = db_manager.join_search(
        target_model=SampleTable,
        join_model=ConditioningTable,
        on_condition=on_condition,
    )
    assert len(results) == 2
