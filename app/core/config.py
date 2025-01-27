import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "bee_vision_db"

    class Config:
        env_file = ".env"

settings = Settings()
