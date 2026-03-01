import os
import pickle
from deepface import DeepFace

dataset_path = "dataset"
embeddings = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        try:
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            embeddings.append({
                "name": person_name,
                "embedding": embedding
            })

            print(f"Encoded: {person_name} - {image_name}")

        except Exception as e:
            print("Error:", e)

with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("All faces encoded successfully!")