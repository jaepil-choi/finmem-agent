from pymongo import MongoClient
from datetime import datetime
import logging

class MongoDBClient:
    def __init__(self, host="localhost", port=27017, db_name="reports"):
        self.uri = f"mongodb://{host}:{port}/"
        self.client = MongoClient(self.uri)
        self.db = self.client[db_name]
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def ensure_timeseries_collections(self, collection_names):
        existing_collections = self.db.list_collection_names()
        for name in collection_names:
            if name not in existing_collections:
                self.logger.info(f"Creating time-series collection: {name}")
                self.db.create_collection(
                    name,
                    timeseries={
                        "timeField": "date",
                        "metaField": "metadata",
                        "granularity": "days"
                    }
                )
            else:
                self.logger.info(f"Collection {name} already exists.")

    def insert_report(self, collection_name, date, filename, text, metadata=None):
        if not isinstance(date, datetime):
            raise ValueError("date must be a datetime object")
        
        document = {
            "date": date,  # Time-series timeField
            "filename": filename,
            "full_text": text,
            "metadata": metadata or {},
            "created_at": datetime.now()
        }
        return self.db[collection_name].insert_one(document)

    def close(self):
        self.client.close()
