from fastapi import APIRouter, WebSocket, UploadFile, File, HTTPException
import cv2
import numpy as np
from starlette.websockets import WebSocketDisconnect

router = APIRouter()

@router.websocket("/live-stream")
async def live_stream(websocket: WebSocket):
    print("Attempting to accept WebSocket connection")
    await websocket.accept()
    try:
        while True:
            # קבלת פריים (Frame) מה-Frontend
            data = await websocket.receive_bytes()

            # המרה ל-Numpy Array ו-OpenCV Frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            process_frame(frame)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API להעלאת וידאו
@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    API להעלאת וידאו מה-Frontend
    """
    try:
        # קריאת הקובץ שהועלה
        video_bytes = await file.read()

        # שמירת הקובץ לדיסק (או המשך עיבוד)
        with open(f"./uploaded_videos/{file.filename}", "wb") as f:
            f.write(video_bytes)
            print("Uploaded video")

        return {"status": "success", "message": "Video uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def process_frame(frame):
    """
    פונקציה לעיבוד פריימים (לדוגמה: זיהוי דבורים מסומנות)
    """
    print("streaming")

