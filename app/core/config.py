import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    MONGO_URI = os.environ.get("MONGO_URI")
    MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME')
    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
    AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
    AWS_S3_REGION = os.environ.get("AWS_S3_REGION")


    class Config:
        env_file = ".env"

settings = Settings()
