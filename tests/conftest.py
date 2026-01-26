import pytest
from pymongo import MongoClient
from src.config import settings

@pytest.fixture(scope="session")
def mongo_client():
    client = MongoClient(settings.MONGODB_HOST, settings.MONGODB_PORT)
    yield client
    client.close()

@pytest.fixture(scope="function")
def test_db(mongo_client):
    db_name = "test_finmem_db"
    # Clean up before test
    mongo_client.drop_database(db_name)
    db = mongo_client[db_name]
    yield db
    # Clean up after test
    mongo_client.drop_database(db_name)
