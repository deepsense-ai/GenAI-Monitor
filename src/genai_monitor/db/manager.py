from typing import Any, Dict, Optional, Sequence, Type, Union

from attrs import define
from loguru import logger
from sqlalchemy import Row, and_

from genai_monitor.db.config import SessionManager
from genai_monitor.db.schemas.base import BaseModel


@define
class DBManager:
    """A database manager class.

    Class provides basic operations for saving, updating and searching records
    in the database using SQLAlchemy ORM models.
    """
    session_manager: SessionManager = None

    def save(self, instance: BaseModel) -> BaseModel:
        """Saves an instance of a model to the database.

        Args:
            instance: The ORM model instance to be saved.

        Returns:
            The saved ORM model instance.
        """
        with self.session_manager.session_scope() as session:
            session.add(instance)
            session.commit()
            self._eager_load_instance_relations(instance=instance)
            return instance

    def search(self, model: Type[BaseModel], filters: Optional[Dict[str, Any]] = None) -> Sequence[Row]:
        """Searches for records in the database that match the given filters.

        Args:
            model: The ORM model class representing the database table to search.
            filters: Dictionary of filter criteria to locate specific records.

        Returns:
            A sequence of rows matching the filter criteria.
        """
        with self.session_manager.session_scope() as session:
            query = session.query(model)
            if filters:
                query = query.filter_by(**filters)
            result = query.all()
            for instance in result:
                self._eager_load_instance_relations(instance)
            return result

    def update(
        self,
        instance: BaseModel = None,
        model: Optional[Type[BaseModel]] = None,
        filters: Optional[Dict[str, Any]] = None,
        values: Optional[Dict[str, Any]] = None,
    ) -> Union[BaseModel, Sequence[Row]]:
        """Updates records in the database.

        If an instance is provided, it will be updated
        directly. Otherwise, model and filter criteria are used to locate records for updating.

        Args:
            instance: An existing ORM model instance to update.
            model: The ORM model class representing the table to update.
            filters: Dictionary of filter criteria to locate records to update.
            values: Dictionary of field names and values to update.

        Returns:
           The updated instance if `instance` was provided or the updated DB rows.

        Raises:
            ValueError: If neither `instance` nor `model` is provided.
        """
        if (instance is None) and (model is None):
            raise ValueError(
                "To update DB resource provide either an instance of the ORM class or an ORM model with filters."
            )

        if (instance is not None) and (model is not None):
            logger.warning(
                "Provided both instance of an ORM class and the ORM model for update, instance will be used."
            )

        with self.session_manager.session_scope() as session:
            if instance is not None:
                for field_name, field_value in values.items(): # type: ignore
                    setattr(instance, field_name, field_value)
                session.add(instance)
                session.commit()
                return instance

            query = session.query(model).filter_by(**filters)
            query_results = query.all()
            for result in query_results:
                for field_name, field_value in values.items(): # type: ignore
                    setattr(result, field_name, field_value)
            session.commit()
            for result in query_results:
                self._eager_load_instance_relations(result)
            return query_results

    def join_search(
        self,
        target_model: Type[BaseModel],
        join_model: Type[BaseModel],
        on_condition: Any,
        target_filters: Optional[Dict[str, Any]] = None,
        join_filters: Optional[Dict[str, Any]] = None,
    ) -> Sequence[Row]:
        """Performs a join search between two models based on a join condition and optional filters.

        Args:
            target_model: The ORM model class representing the primary table to search.
            join_model: The ORM model class representing the table to join with.
            on_condition: The join condition specifying how to link the two tables.
            target_filters: Dictionary of filter criteria for the target model.
            join_filters: Dictionary of filter criteria for the join model.

        Returns:
            A sequence of rows resulting from the join search, filtered as specified.
        """
        with self.session_manager.session_scope() as session:
            query = session.query(target_model).join(join_model, on_condition)

            if target_filters:
                target_conditions = [getattr(target_model, key) == value for key, value in target_filters.items()]
                query = query.filter(and_(*target_conditions))

            if join_filters:
                join_conditions = [getattr(join_model, key) == value for key, value in join_filters.items()]
                query = query.filter(and_(*join_conditions))

            results = query.all()
            return results

    @staticmethod
    def _eager_load_instance_relations(instance: BaseModel): # noqa: ANN205
        for relation in instance.relations:
            getattr(instance, relation)
