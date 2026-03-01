import cv2
from deepface import DeepFace
from app.config import CAMERA_INDEX, MODEL_NAME
from app.recognizer import recognize_face
from app.utils import similarity_to_confidence

def start_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = DeepFace.represent(
                img_path=frame,
                model_name=MODEL_NAME,
                enforce_detection=False
            )

            for face in results:
                embedding = face["embedding"]

                name, similarity = recognize_face(embedding)
                confidence = similarity_to_confidence(similarity)

                label = f"{name} ({confidence}%)"

                cv2.putText(frame, label, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

        except:
            pass

        cv2.imshow("Face Recognition System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()