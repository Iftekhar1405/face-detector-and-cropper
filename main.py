from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
import numpy as np
import tempfile
import zipfile
import os

app = FastAPI()

@app.post("/crop-faces/")
async def crop_faces(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if len(faces) == 0:
        return {"error": "No faces detected"}

    # Case 1: Single face → return directly
    if len(faces) == 1:
        (x, y, w, h) = faces[0]

        padding = 80  # adjust this value as needed

        # Apply padding but keep within image bounds
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        face = img[y1:y2, x1:x2]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            cv2.imwrite(tmpfile.name, face)
            return FileResponse(tmpfile.name, media_type="image/png", filename="face.png")

    # Case 2: Multiple faces → return ZIP
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "cropped_faces.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y+h, x:x+w]
                face_path = os.path.join(tmpdir, f"face_{i+1}.png")
                cv2.imwrite(face_path, face)
                zipf.write(face_path, f"face_{i+1}.png")

        return FileResponse(zip_path, media_type="application/zip", filename="cropped_faces.zip")
