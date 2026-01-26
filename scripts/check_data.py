from pymongo import MongoClient
from src.config import settings

def check_data():
    client = MongoClient(settings.MONGODB_HOST, settings.MONGODB_PORT)
    db = client[settings.MONGODB_DB_NAME]
    
    print("--- MongoDB Collections ---")
    for col in ["daily", "weekly", "monthly"]:
        count = db[col].count_documents({})
        print(f"{col}: {count} documents")
        if count > 0:
            latest = db[col].find_one(sort=[("date", -1)])
            print(f"  Latest date in {col}: {latest.get('date')}")

if __name__ == "__main__":
    check_data()
