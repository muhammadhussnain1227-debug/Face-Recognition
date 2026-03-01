import os
import pickle
from deepface import DeepFace
from app.config import DATASET_PATH, EMBEDDINGS_PATH, MODEL_NAME

def encode_faces():
    embeddings = []

    for person_name in os.listdir(DATASET_PATH):
        person_folder = os.path.join(DATASET_PATH, person_name)

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            try:
                result = DeepFace.represent(
                    img_path=image_path,
                    model_name=MODEL_NAME,
                    enforce_detection=False
                )

                embedding = result[0]["embedding"]

                embeddings.append({
                    "name": person_name,
                    "embedding": embedding
                })

                print(f"Encoded {person_name} - {image_name}")

            except Exception as e:
                print("Error:", e)

    os.makedirs("models", exist_ok=True)

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    print("All faces encoded successfully!")