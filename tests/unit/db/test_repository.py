import pytest
from datetime import datetime, time
from src.db.repository import ReportRepository

@pytest.mark.usefixtures("test_db")
class TestReportRepository:
    @pytest.fixture(autouse=True)
    def setup(self, test_db):
        self.db = test_db
        # We need to ensure the MongoDBClient uses the test DB
        from src.db.mongodb_client import MongoDBClient
        db_client = MongoDBClient(db_name=test_db.name)
        self.repo = ReportRepository(db_client=db_client)

    def test_get_reports_by_date_real_db(self):
        # 1. Prepare data
        target_date = datetime(2024, 1, 1, 10, 0)
        other_date = datetime(2024, 1, 2, 10, 0)
        
        # Reports on the target date
        self.repo.client.insert_report("daily", target_date, "daily1.txt", "content1")
        self.repo.client.insert_report("weekly", target_date, "weekly1.txt", "content2")
        
        # Report on a different date
        self.repo.client.insert_report("daily", other_date, "daily2.txt", "content3")

        # 2. Query
        results = self.repo.get_reports_by_date(target_date)

        # 3. Verify
        assert len(results) == 2
        filenames = [r["filename"] for r in results]
        assert "daily1.txt" in filenames
        assert "weekly1.txt" in filenames
        assert "daily2.txt" not in filenames

        for r in results:
            assert r["date"].date() == target_date.date()
