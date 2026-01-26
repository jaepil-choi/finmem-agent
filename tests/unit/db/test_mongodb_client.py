import pytest
from datetime import datetime
from src.db.mongodb_client import MongoDBClient

@pytest.mark.usefixtures("test_db")
class TestMongoDBClient:
    @pytest.fixture(autouse=True)
    def setup(self, test_db):
        self.db = test_db
        self.db_name = test_db.name
        self.client = MongoDBClient(db_name=self.db_name)

    def test_insert_report_validates_date(self):
        # Should raise ValueError if date is not a datetime object
        with pytest.raises(ValueError):
            self.client.insert_report("collection", "not-a-date", "file.txt", "content")

    def test_insert_report_real_db(self):
        collection_name = "test_col"
        test_date = datetime(2024, 1, 1)
        
        # Insert using our client
        self.client.insert_report(collection_name, test_date, "file.txt", "content")
        
        # Verify in real DB
        doc = self.db[collection_name].find_one({"filename": "file.txt"})
        assert doc is not None
        assert doc["date"] == test_date
        assert doc["full_text"] == "content"
        assert "created_at" in doc
        assert isinstance(doc["created_at"], datetime)
