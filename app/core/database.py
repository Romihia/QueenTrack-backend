from motor.motor_asyncio import AsyncIOMotorClient
from .config import settings

# יצירת Client גלובלי לחיבור ל-Mongo
client = AsyncIOMotorClient(settings.MONGO_URI)
db = client[settings.MONGO_DB_NAME]
