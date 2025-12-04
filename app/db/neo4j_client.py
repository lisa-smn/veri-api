from neo4j import GraphDatabase, Driver, Session
from app.core.config import settings

_driver: Driver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_url,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


def get_neo4j_session() -> Session:
    return get_driver().session()


def close_driver():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
