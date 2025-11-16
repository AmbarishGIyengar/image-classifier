# image-classifier
Object Detection API using YOLOv8 and FastAPI
Project Overview
This project implements a REST API for real-time object detection on images using the YOLOv8 computer vision model. The API accepts image uploads and returns detected objects with bounding boxes, class labels, and confidence scores.

Key technologies:

YOLOv8: State-of-the-art object detection model.

FastAPI: Fast, asynchronous web framework for building APIs.

OpenCV: Used for image processing.

Python: Core programming language.

Features
Upload images via a single /detect POST endpoint.

Real-time detection using a pretrained YOLOv8 model.

Returns JSON response with objects detected, bounding box coordinates, and confidence.

Lazy loads the YOLOv8 model to improve API startup time.

Exception handling to gracefully handle errors and send informative responses.

Getting Started
Prerequisites
Python 3.8+

Install required Python packages:

bash
pip install fastapi uvicorn opencv-python ultralytics numpy
Running the API
Clone the repository or download the project files.

Run the FastAPI server:

bash
uvicorn app:app --reload
Access the interactive API docs in your browser at:

text
http://127.0.0.1:8000/docs
Use the /detect endpoint to upload images and get JSON results.

Usage Example
The /detect endpoint accepts multipart form data with an image file.

The response JSON contains a list of detected objects with class names, bounding box coordinates, and confidence scores.

Sample response:

json
{
  "detections": [
    {
      "class": "person",
      "confidence": 0.98,
      "box": [100.5, 200.2, 300.3, 500.7]
    },
    {
      "class": "bicycle",
      "confidence": 0.87,
      "box": [400.1, 300.4, 600.2, 500.8]
    }
  ]
}
Project Structure
app.py: Main FastAPI app with the detect endpoint.

YOLOv8 model is lazy-loaded for efficiency.

Exception handling included to manage errors during image processing and inference.

Next Steps and Enhancements
Add endpoint to return annotated images with bounding boxes drawn.

Add authentication and rate-limiting.

Integrate with a front-end interface.

Deploy API using Docker and cloud services.
