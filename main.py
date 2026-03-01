from app.face_encoder import encode_faces
from app.camera import start_camera
import os

if not os.path.exists("models/embeddings.pkl"):
    print("Encoding faces...")
    encode_faces()

print("Starting camera...")
start_camera()