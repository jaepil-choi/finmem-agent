from datetime import datetime, time
from typing import List, Dict, Any, Optional
from src.db.mongodb_client import MongoDBClient

class ReportRepository:
    def __init__(self, db_client: Optional[MongoDBClient] = None):
        self.client = db_client or MongoDBClient()

    def get_reports_by_date(self, target_date: datetime, collections: List[str] = ["daily", "weekly", "monthly"]) -> List[Dict[str, Any]]:
        """
        Fetches all reports from the specified collections that were published on the target_date.
        """
        # Define start and end of the target day
        start_of_day = datetime.combine(target_date.date(), time.min)
        end_of_day = datetime.combine(target_date.date(), time.max)

        all_reports = []
        for col_name in collections:
            # Time-series collections in MongoDB store 'date' as a BSON date
            query = {
                "date": {
                    "$gte": start_of_day,
                    "$lte": end_of_day
                }
            }
            cursor = self.client.db[col_name].find(query)
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                doc["category"] = col_name
                all_reports.append(doc)
        
        return all_reports

    def close(self):
        self.client.close()
