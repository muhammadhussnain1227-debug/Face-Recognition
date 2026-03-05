# Real-Time Face Recognition System

A **real-time face recognition system** built using **Python, OpenCV, and DeepFace (FaceNet model)**.

The system encodes faces from a dataset, stores embeddings, and performs **live face detection and recognition using a webcam**.

It also includes a **premium UI overlay, confidence scoring, and automatic face counting**.

---

# Features

- Real-time face detection and recognition
- FaceNet embeddings for accurate recognition
- Cosine-distance based similarity matching
- Confidence score display
- Real-time face counter
- Dynamic colored bounding boxes
- Stylish UI overlay panel
- Automatic dataset encoding
- Unknown face detection
- Face image renaming utility

---

# Project Structure

```
Face-Recognition-System
│
├── dataset/
│   ├── Person1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │
│   ├── Person2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│
├── encode_faces.py
├── realtime_recognition.py
├── rename.py
├── embeddings.pkl
├── requirements.txt
└── README.md
```

---

# Installation

## 1 Install Python Dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```
deepface
opencv-python
numpy
```

---

# Dataset Preparation

Create a **dataset folder** where each person has their own subfolder.

Example:

```
dataset/
│
├── Ahmad/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── Zia/
│   ├── img1.jpg
│   ├── img2.jpg
```

Each **folder name will be used as the identity label**.

---

# Step 1: Rename Dataset Images (Optional)

Use `rename.py` to automatically rename images inside a person's folder.

Example configuration:

```python
folder_path = r"C:\Users\User\Desktop\Projects\Face recognition\dataset\Mohid Hussain"
person_name = "Mohid Hussain"
```

Run:

```bash
python rename.py
```

Output example:

```
Mohid Hussain_1.jpg
Mohid Hussain_2.jpg
Mohid Hussain_3.jpg
```

---

# Step 2: Encode Faces

The script `encode_faces.py` extracts **FaceNet embeddings** from the dataset and stores them in a pickle file.

Run:

```bash
python encode_faces.py
```

Output:

```
Encoded: Ahmad - img1.jpg
Encoded: Ahmad - img2.jpg
Encoded: Zia - img1.jpg
Encoded: Zia - img2.jpg

All faces encoded successfully!
```

This generates:

```
embeddings.pkl
```

This file contains **all face embeddings used for recognition**.

---

# Step 3: Run Real-Time Recognition

Run the webcam recognition system.

```bash
python realtime_recognition.py
```

The system will:

- Open webcam
- Detect faces
- Extract embeddings
- Compare with stored embeddings
- Display **name + confidence**

Press **Q** to exit.

---

# Recognition Pipeline

The system follows this pipeline:

```
Dataset Images
      │
      ▼
DeepFace FaceNet Model
      │
      ▼
Face Embeddings
      │
      ▼
Store in embeddings.pkl
      │
      ▼
Webcam Frame
      │
      ▼
Face Detection
      │
      ▼
Embedding Extraction
      │
      ▼
Distance Comparison
      │
      ▼
Identity Prediction
```

---

# Matching Algorithm

The system uses **Euclidean Distance** between normalized embeddings.

```
distance = ||embedding1 - embedding2||
```

Confidence is calculated as:

```
confidence = 1 - distance
```

If distance is greater than threshold:

```
Identity = Unknown
```

---

# Real-Time UI Features

## Face Detection

- Bounding boxes drawn around detected faces
- Each identity gets a unique color

---

## Identity Display

Example label:

```
Ahmad (0.87)
```

Where:

- **Ahmad** = recognized identity
- **0.87** = confidence score

---

## Face Counter Panel

A **premium overlay panel** shows the number of faces in the frame.

Example:

```
Total Faces: 3
```

Panel colors:

- 🟢 Green → Faces detected  
- 🔴 Red → No faces detected  

---

# Example Output

```
┌─────────────────────────┐
│ Total Faces: 2          │
└─────────────────────────┘

[ Ahmad (0.91) ]   [ Zia (0.88) ]
```

---

# Technologies Used

| Technology | Purpose |
|-----------|--------|
| Python | Core programming |
| DeepFace | Face embeddings |
| FaceNet | Feature extraction |
| OpenCV | Webcam + image processing |
| NumPy | Numerical computations |
| Pickle | Embedding storage |

---

# Future Improvements

Possible upgrades:

- Face database using **SQLite / PostgreSQL**
- Face attendance system
- Multi-camera support
- Face mask detection
- Face tracking (DeepSORT)
- Web dashboard
- REST API using **FastAPI**
- Docker containerization
- GPU acceleration

---

# Known Limitations

- Recognition accuracy depends on **dataset quality**
- Low lighting can reduce detection performance
- Similar faces may require more training images

---

# Author

**Muhammad Hussnain**

AI / Computer Vision Enthusiast  
Focused on **AI systems, real-time recognition, and ML engineering**