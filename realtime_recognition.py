import cv2
import pickle
import numpy as np
from deepface import DeepFace

# Load stored embeddings
with open("embeddings.pkl", "rb") as f:
    stored_embeddings = pickle.load(f)


def find_match(embedding, threshold=0.9):
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


# Assign unique colors
color_map = {}

def get_color(name):
    if name not in color_map:
        color_map[name] = tuple(np.random.randint(50, 255, 3).tolist())
    return color_map[name]


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

            x = face["facial_area"]["x"]
            y = face["facial_area"]["y"]
            w = face["facial_area"]["w"]
            h = face["facial_area"]["h"]

            color = get_color(name)

            # Premium bounding box (thicker + smooth)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

            label = f"{name}  {confidence:.2f}"

            # Text background box
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )

            cv2.rectangle(frame,
                          (x, y - th - 15),
                          (x + tw + 10, y),
                          color, -1)

            cv2.putText(frame,
                        label,
                        (x + 5, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2)

        # ===== PREMIUM TOTAL FACE PANEL =====
        overlay = frame.copy()

        # Glass effect background
        cv2.rectangle(overlay, (15, 15), (320, 75), (20, 20, 20), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Neon border
        cv2.rectangle(frame, (15, 15), (320, 75), (0, 255, 150), 2)

        cv2.putText(frame,
                    f"Total Faces: {face_count}",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 150),
                    2)

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()