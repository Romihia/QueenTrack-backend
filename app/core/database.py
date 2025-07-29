from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from .config import settings
import logging

logger = logging.getLogger(__name__)

# יצירת Client גלובלי לחיבור ל-Mongo
client = AsyncIOMotorClient(settings.MONGO_URI)
db = client[settings.MONGO_DB_NAME]

async def get_database() -> AsyncIOMotorDatabase:
    """Get database instance"""
    try:
        # Test connection
        await client.admin.command('ping')
        logger.info("✅ Database connection successful")
        return db
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise

async def init_db():
    """Initialize database connection"""
    try:
        database = await get_database()
        logger.info(f"📊 Connected to MongoDB: {settings.MONGO_DB_NAME}")
        return database
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise
