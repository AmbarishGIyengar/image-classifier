from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO('yolov8n.pt')
def load_model():
    global model
    if model is None:
        model = YOLO('yolov8n.pt')
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        load_model()  # Lazy load on first inference request

        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(content={"error": "image not valid"}, status_code=400)

        results = model(img)
        result = results[0]

        boxes = result.boxes
        class_names = model.names

        detections = []
        for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "class": class_names[int(cls)],
                "confidence": float(conf),
                "box": [x1, y1, x2, y2]
            })

        return {"detections": detections}

    except Exception as e:
        print(f"Error during detection: {e}")
        return JSONResponse(content={"error": "Internal server error", "message": str(e)}, status_code=500)
@app.get("/health/")
async def health_check():
    return {"status": "ok"}