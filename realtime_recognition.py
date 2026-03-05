import cv2
import pickle
import numpy as np
from deepface import DeepFace

# Load stored embeddings
with open("embeddings.pkl", "rb") as f:
    stored_embeddings = pickle.load(f)

def find_match(embedding, threshold=1.1):
    min_distance = float("inf")
    identity = "Unknown"

    embedding = embedding / np.linalg.norm(embedding)

    for data in stored_embeddings:
        stored_embedding = np.array(data["embedding"])
        stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)

        distance = np.linalg.norm(stored_embedding - embedding)

        if distance < min_distance:
            min_distance = distance
            identity = data["name"]

    confidence = max(0, 1 - min_distance)

    if min_distance < threshold:
        return identity, confidence
    else:
        return "Unknown", confidence

# Assign unique colors for identities
color_map = {}
def get_color(name):
    if name not in color_map:
        color_map[name] = tuple(np.random.randint(50, 255, 3).tolist())
    return color_map[name]

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not working!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces with bounding boxes using DeepFace
        results = DeepFace.represent(
            img_path=frame_rgb,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="opencv"
        )

        face_count = len(results)

        for face in results:

            embedding = np.array(face["embedding"])
            name, confidence = find_match(embedding)

            # Get bounding box
            x = face["facial_area"]["x"]
            y = face["facial_area"]["y"]
            w = face["facial_area"]["w"]
            h = face["facial_area"]["h"]

            color = get_color(name)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    except Exception as e:
        # If DeepFace fails, face_count = 0
        face_count = 0
        # print("Error:", e)

    # ======================
    # Premium overlay panel (0-face logic + stylish UI)
    # ======================
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 70), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    panel_color = (0, 0, 255) if face_count == 0 else (0, 255, 150)
    cv2.rectangle(frame, (10, 10), (300, 70), panel_color, 2)

    cv2.putText(frame, f"Total Faces: {face_count}", (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, panel_color, 2)

    # Show frame
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()