from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import cv2
import numpy as np
import requests
import tempfile

app = FastAPI()

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.get("/crop-face")
def crop_face_from_url(url: str):
    try:
        # Fetch the image from URL
        resp = requests.get(url, stream=True, timeout=10)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image from URL")

        # Convert to np array
        img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            raise HTTPException(status_code=404, detail="No faces detected")

        # Crop the first face (you can loop for multiple if needed)
        (x, y, w, h) = faces[0]

        # Optional padding
        pad = 80
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = w + pad * 2
        h = h + pad * 2

        cropped = img[y:y+h, x:x+w]

        # Save temp file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(tmp_file.name, cropped)

        return FileResponse(tmp_file.name, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
