import cv2
import pickle
import numpy as np
from deepface import DeepFace

# Load stored embeddings
with open("embeddings.pkl", "rb") as f:
    stored_embeddings = pickle.load(f)

def find_match(embedding, threshold=0.9):
    """
    Compare the given embedding with stored embeddings
    using L2-normalized Euclidean distance.
    """
    min_distance = float("inf")
    identity = "Unknown"

    # Normalize the input embedding
    embedding = embedding / np.linalg.norm(embedding)

    for data in stored_embeddings:
        stored_embedding = np.array(data["embedding"])
        stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)

        distance = np.linalg.norm(stored_embedding - embedding)

        if distance < min_distance:
            min_distance = distance
            identity = data["name"]

    if min_distance < threshold:
        return identity
    else:
        return "Unknown"

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Convert frame to RGB for DeepFace
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face and get embeddings
        results = DeepFace.represent(
            img_path=frame_rgb,
            model_name="Facenet",
            enforce_detection=False
        )

        # Handle multiple faces (if any)
        for face in results:
            embedding = np.array(face["embedding"])
            name = find_match(embedding, threshold=0.9)

            # Draw label on frame
            cv2.putText(frame, name, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    except Exception as e:
        # Debug errors
        print("Error:", e)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()