# Biometric Attendance System - Technical Documentation

## 1. Project Overview

### 1.1 System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Biometric Attendance System              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │   Dataset   │───>│   Training  │───>│  Encodings.pkl  │ │
│  │  (Images)   │    │   Module    │    │  (128-D Vectors)│ │
│  └─────────────┘    └─────────────┘    └─────────────────┘ │
│                                                          │ │
│  ┌────────────────────────────────────────────────────────┐│
│  │              Real-time Processing Pipeline             ││
│  │  ┌──────────┐ → ┌──────────┐ → ┌──────────┐ → ┌──────┐ ││
│  │  │  Video   │   │  Face    │   │  Face    │   │ Track  │ ││
│  │  │ Capture  │   │ Detection│   │Recognition│   │ & Mark│ ││
│  │  │ (OpenCV) │   │  (dlib)  │   │(sklearn) │   │Attendance│
│  │  └──────────┘   └──────────┘   └──────────┘   └──────┘ ││
│  └────────────────────────────────────────────────────────┘│
│                             │                              │
│                             ▼                              │
│  ┌────────────────────────────────────────────────────────┐│
│  │              Data Persistence Layer                    ││
│  │  ┌─────────────┐      ┌─────────────┐                 ││
│  │  │  SQLite DB  │      │  CSV Report │                 ││
│  │  │(attendance) │      │ (export)    │                 ││
│  │  └─────────────┘      └─────────────┘                 ││
│  └────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Tech Stack
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.x | Core implementation |
| **Computer Vision** | OpenCV | Video capture, frame processing |
| **Face Detection** | dlib (HOG/CNN) | Face localization in frames |
| **Face Recognition** | face_recognition | 128-D face encoding generation |
| **ML/Search** | scikit-learn (NearestNeighbors) | Fast similarity matching |
| **Database** | SQLite | Attendance persistence |
| **Data Analysis** | pandas | Report generation |
| **Utilities** | numpy, tqdm | Array operations, progress bars |

---

## 2. Core Algorithms & Technical Concepts

### 2.1 Face Recognition Pipeline (dlib + face_recognition)

#### Step 1: Face Detection (HOG vs CNN)
```python
# HOG (Histogram of Oriented Gradients)
- Faster but less accurate (~5-10 FPS)
- CPU-based
- Good for real-time applications

# CNN (Convolutional Neural Network)  
- More accurate, slower (~2-5 FPS)
- Can use GPU if available
- Better for training phase
```

**Implementation**: `@src/face_detector.py:24`
```python
face_recognition.face_locations(rgb_frame, model=self.model)
# model="hog" or "cnn"
```

#### Step 2: Face Encoding (128-D Vector Generation)
- Uses pre-trained ResNet model (from dlib)
- Generates 128-dimensional face embedding
- Same person → similar vectors (Euclidean distance < threshold)

**Implementation**: `@src/face_detector.py:35-39`
```python
face_recognition.face_encodings(
    image,
    known_face_locations=face_locations,
    num_jitters=self.num_jitters  # Sampling variations for accuracy
)
```

#### Step 3: Recognition via Nearest Neighbors
- **Algorithm**: Euclidean distance-based Nearest Neighbors
- **Threshold**: Default 0.5 (configurable)
- **Library**: scikit-learn NearestNeighbors

**Implementation**: `@src/recognizer.py:59-62`
```python
self.nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean")
self.nn_model.fit(self.known_encodings)  # Build search index

# Query time: O(log n) via KD-Tree or Ball-Tree
```

**Distance-to-Confidence Conversion**:
```python
confidence = max(0.0, 1.0 - (distance / threshold))
# distance=0.0 → confidence=100%
# distance=0.5 → confidence=0% (at threshold)
```

---

### 2.2 Object Tracking (Centroid Tracker)

**Purpose**: Maintain identity of detected faces across consecutive frames without re-running expensive recognition on every frame.

**Algorithm** (`@src/video_processor.py:31-134`):

1. **Centroid Calculation**:
   ```python
   c_x = (left + right) / 2
   c_y = (top + bottom) / 2
   ```

2. **Distance Matrix** (Hungarian algorithm concept):
   ```python
   distance_matrix = np.linalg.norm(
       object_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :],
       axis=2
   )
   ```

3. **Track Assignment**:
   - Match existing tracks to new detections based on minimum centroid distance
   - Max distance threshold: 80 pixels (configurable)
   - Max disappeared frames: 20 (configurable)

4. **Recognition Cooldown**:
   ```python
   if frame_count - track.last_recognition_frame < cooldown_frames:
       skip_recognition  # Avoid redundant computation
   ```

---

### 2.3 Attendance Decision Logic

**Present/Absent Threshold**: 75% detection rate during session

**Implementation**: `@src/video_processor.py:314-326`
```python
detection_threshold = 0.75  # 75% of detection attempts

for name in db_students:
    detections = successful_detections_per_student.get(name, 0)
    percent = (detections / total_detection_attempts * 100)
    status = "Present" if percent >= 75.0 else "Absent"
```

**Key Insight**: A student is marked "Present" only if their face is successfully recognized in ≥75% of the processed frames during the session. This prevents false positives from sporadic detections.

---

## 3. Data Storage Schema

### 3.1 SQLite Database Structure

**Students Table** (`@src/attendance_manager.py:27-30`):
```sql
CREATE TABLE students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);
```

**Attendance Table** (`@src/attendance_manager.py:36-47`):
```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,          -- Student identifier
    date TEXT NOT NULL,          -- YYYY-MM-DD
    time TEXT NOT NULL,          -- HH:MM:SS
    status TEXT NOT NULL,        -- Present/Absent
    session TEXT,                -- Custom session name
    duration REAL,               -- Detection percentage
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### 3.2 Encodings File (.pkl)
Pickle-serialized dictionary:
```python
{
    "encodings": np.ndarray,  # Shape: (n_samples, 128), float32
    "names": List[str],       # Student name per encoding
    "metadata": {
        "model": "hog"/"cnn",
        "num_jitters": int,
        "students": List[str],
        "total_samples": int,
        "embedding_size": 128
    }
}
```

---

## 4. Performance Optimizations

### 4.1 Frame Processing Pipeline
```python
# 1. Resize frame before detection (25% scale default)
rgb_small = cv2.resize(frame_bgr, (0, 0), fx=0.25, fy=0.25)

# 2. Process every Nth frame (skip frames, default every 2nd)
if frame_count % process_every_n == 0:
    run_detection_and_recognition()

# 3. Scale locations back to original frame for display
locations_full = locations_small * 4  # inverse of 0.25
```

### 4.2 Computational Savings
| Optimization | Impact |
|-------------|--------|
| Frame resize (0.25x) | **16x fewer pixels** to process |
| Skip frames (every 2nd) | **2x fewer detection calls** |
| Recognition cooldown | Avoid redundant face recognition on same track |
| Centroid tracker | Maintain identity without re-recognition |

---

## 5. Key Configuration Parameters

| Parameter | Default | Description | Location |
|-----------|---------|-------------|----------|
| `threshold` | 0.5 | Face recognition distance threshold | `run_attendance.py:25` |
| `detection-model` | hog | Face detection backend (hog/cnn) | `run_attendance.py:29` |
| `process-every-n` | 2 | Frame skip factor | `run_attendance.py:36` |
| `resize-scale` | 0.25 | Frame resize factor | `run_attendance.py:42` |
| `recognition-cooldown` | 30 | Frames before re-recognition | `run_attendance.py:48` |
| `tracker-max-disappeared` | 20 | Frames before track deletion | `run_attendance.py:54` |
| `tracker-max-distance` | 80.0 | Max centroid distance for matching | `run_attendance.py:60` |

---

## 6. Execution Flow

### 6.1 Training Phase (One-time setup)
```bash
python train_model.py --dataset-dir ./dataset --encodings-path ./encodings/encodings.pkl
```

**Flow**:
1. Load images from `dataset/<student_name>/<images>`
2. Detect face in each image (largest face if multiple)
3. Generate 128-D encoding
4. Build NearestNeighbors index
5. Save to pickle file

### 6.2 Runtime Phase (Attendance Session)
```bash
python run_attendance.py --source 0 --threshold 0.5
```

**Flow**:
1. Load encodings from `.pkl` file
2. Initialize SQLite database
3. Start video capture (webcam/file)
4. For each frame:
   - Preprocess (resize, color convert)
   - Detect faces (every Nth frame)
   - Track faces (centroid tracker)
   - Recognize new tracks (NN search)
   - Draw bounding boxes + labels
5. On session end: Calculate detection %, mark attendance
6. Generate CSV report

---

## 7. Interview/Viva Talking Points

### Q: "Why 128-D embeddings?"
**A**: The dlib face recognition model uses a ResNet-based CNN trained to produce 128-dimensional face descriptors. This dimensionality strikes a balance between discriminative power (distinguishing different faces) and computational efficiency. Research shows this provides near-human-level accuracy (99.38% on LFW benchmark).

### Q: "Why Euclidean distance vs Cosine similarity?"
**A**: Face embeddings from dlib are already L2-normalized during training, making Euclidean distance equivalent to cosine distance (for unit vectors: ||a-b||² = 2 - 2cosθ). Euclidean is intuitive and efficient with scikit-learn's KD-Tree implementation.

### Q: "How does the tracker improve performance?"
**A**: The centroid tracker maintains face identity across frames using spatial continuity. This avoids re-running expensive face recognition on every frame of the same person. Only new/unknown tracks trigger recognition, reducing computation by ~80% in typical classroom scenarios.

### Q: "What is num_jitters and why does it matter?"
**A**: `num_jitters` controls how many random perturbations (slight rotations, scales) are applied during encoding generation. Higher values = more accurate but slower. Default of 1 provides good speed-accuracy tradeoff for real-time use.

### Q: "Why 75% detection threshold for Present/Absent?"
**A**: This prevents marking someone "Present" due to a single accidental detection (e.g., walking past the camera). Requires sustained presence during the session. Configurable based on session duration and environment.

### Q: "How would you scale this to 1000+ students?"
**A**: 
1. Replace NearestNeighbors with FAISS (Facebook AI Similarity Search) for million-scale vector search
2. Use approximate nearest neighbors (ANN) for sub-linear query time
3. Shard database by class/department
4. Implement caching for frequently seen students

---

## 8. Project Structure

```
Biometric_Attendance/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── face_detector.py         # Detection + encoding (dlib wrapper)
│   ├── face_encoder.py          # Training pipeline
│   ├── recognizer.py            # NearestNeighbors recognition
│   ├── video_processor.py       # Main runtime + tracking
│   ├── attendance_manager.py    # Database operations
│   └── utils.py                 # Path configs, logging helpers
├── run_attendance.py            # Entry point for attendance
├── train_model.py               # Entry point for training
├── requirements.txt             # Dependencies
├── dataset/                     # Training images (student folders)
├── encodings/                   # Generated .pkl files
├── database/                  
│   ├── attendance.db            # SQLite database
│   └── view_attendance.py       # DB viewer utility
└── reports/
    └── attendance_report.csv    # Exported attendance data
```

---

## 9. Sample Output Interpretation

```
Session 'session2': Varun2022BCY0015 - Absent (58.1% detection)
Session 'session2': Suraj2022BCS0051 - Absent (0.0% detection)
```

**Interpretation**:
- **Varun**: Detected in 58.1% of frames → Below 75% threshold → Marked Absent
- **Suraj**: Never detected → 0.0% → Marked Absent
- Session duration was 20 seconds, processed every 2nd frame

---

## 10. Dependencies Explained

| Package | Role |
|---------|------|
| `opencv-python` | Video I/O, image manipulation |
| `face-recognition` | High-level API over dlib |
| `dlib-bin` | Low-level face detection/encoding (C++ backend) |
| `scikit-learn` | NearestNeighbors for fast matching |
| `pandas` | CSV export, data manipulation |
| `numpy` | Array operations, vector math |
| `tqdm` | Progress bars for training |

---

*Documentation generated for project viva preparation*
