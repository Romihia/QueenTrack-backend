from fastapi import FastAPI
from app.routes.routes import router as events_router

app = FastAPI(
    title="Bee Vision - Event API",
    version="0.1.0"
)

# הגדרת הנתיב /events
app.include_router(events_router, prefix="/events", tags=["events"])

@app.get("/")
def root():
    return {"message": "Bee Vision Backend running."}
