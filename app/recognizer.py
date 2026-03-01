import pickle
import numpy as np
from app.config import EMBEDDINGS_PATH, THRESHOLD
from app.utils import cosine_similarity

with open(EMBEDDINGS_PATH, "rb") as f:
    STORED_EMBEDDINGS = pickle.load(f)

def recognize_face(embedding):
    best_match = "Unknown"
    highest_similarity = 0

    for data in STORED_EMBEDDINGS:
        similarity = cosine_similarity(data["embedding"], embedding)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = data["name"]

    if highest_similarity >= THRESHOLD:
        return best_match, highest_similarity
    else:
        return "Unknown", highest_similarity