import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    MONGO_URI = os.environ.get('MONGO_URI')
    MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME')
    
    # Local video storage configuration
    VIDEOS_DIR = os.environ.get('VIDEOS_DIR', '/data/videos')

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
