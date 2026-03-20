# Automated Student Attendance System from Video Sequence

A production-style Python system that detects and recognizes student faces from a webcam or classroom video, marks attendance automatically in SQLite, and exports attendance reports to CSV.

## Features

- Real-time face detection from webcam/video stream
- Face recognition using pre-trained student embeddings
- Automatic attendance marking with duplicate prevention (once per student per day)
- SQLite database storage for students and attendance records
- CSV report generation
- Performance optimizations:
  - Process every Nth frame
  - Frame resizing to 1/4 before detection
  - Cached face encodings loaded from pickle
  - Fast nearest-neighbor matching (`scikit-learn`)
- Centroid tracking to reduce repeated recognition across frames
- Logging and robust error handling

## Project Structure

```text
attendance_system/
|
|-- dataset/
|   |-- student1/
|   |   |-- img1.jpg
|   |   |-- img2.jpg
|   |-- student2/
|
|-- encodings/
|   |-- encodings.pkl
|
|-- database/
|   |-- attendance.db
|
|-- reports/
|   |-- attendance_report.csv
|
|-- src/
|   |-- face_detector.py
|   |-- face_encoder.py
|   |-- recognizer.py
|   |-- attendance_manager.py
|   |-- video_processor.py
|   |-- utils.py
|
|-- train_model.py
|-- run_attendance.py
|-- requirements.txt
|-- README.md
```

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

2. Install base dependencies:

```bash
pip install -r requirements.txt
```

3. Install `face-recognition`:

Windows (recommended with prebuilt `dlib-bin` from requirements):

```bash
pip install face-recognition==1.3.0 --no-deps
```

Linux/Mac (with native build toolchain installed):

```bash
pip install face-recognition==1.3.0
```

## Add Student Images

1. Create one folder per student inside `dataset/`.
2. Use the folder name as the student identity.
3. Add multiple clear face images per student (recommended: 5 to 20 images).

Example:

```text
dataset/
  Rahul/
    img1.jpg
    img2.jpg
  Priya/
    img1.jpg
    img2.jpg
```

Notes:

- Use front-facing, well-lit images.
- Avoid group photos for training images.
- If multiple faces exist in one image, the largest face is used.

## Train Face Encodings

Run:

```bash
python train_model.py
```

Optional flags:

```bash
python train_model.py --dataset-dir dataset --encodings-path encodings/encodings.pkl --detection-model hog --num-jitters 1
```

This creates `encodings/encodings.pkl`.

## Run Attendance System

Webcam (default index `0`):

```bash
python run_attendance.py
```

Video file input:

```bash
python run_attendance.py --source path/to/classroom_video.mp4
```

Important options:

```bash
python run_attendance.py \
  --threshold 0.5 \
  --process-every-n 2 \
  --resize-scale 0.25 \
  --recognition-cooldown 30 \
  --tracker-max-disappeared 20 \
  --tracker-max-distance 80
```

Press `q` in the video window to stop.

## Database Schema

The system auto-creates `database/attendance.db` with:

- `students(id, name)`
- `attendance(id, name, date, time, status)`

`attendance` has uniqueness constraint on `(name, date)` to prevent duplicate daily entries.

## Attendance Report

A CSV report is generated at:

- `reports/attendance_report.csv`

Columns:

- `Name`
- `Date`
- `Time`
- `Status`

## Example Runtime Logs

```text
Detected: Rahul -> Attendance Marked
Detected: Priya -> Attendance Marked
Unknown face detected
```

## Example Output Screenshots

Add your screenshots to `reports/` and reference them here:

- `reports/screenshot_live_detection.png`
- `reports/screenshot_console_logs.png`

## Troubleshooting

- If `encodings.pkl` is missing, run `python train_model.py` first.
- If camera cannot open, try `--source 1` or close other camera apps.
- If recognition is weak:
  - add more diverse training images per student,
  - improve classroom lighting,
  - decrease threshold slightly (e.g., `0.45`) for stricter matching.
- If `face_recognition_models` complains about `pkg_resources`, keep `setuptools<81` (already pinned in `requirements.txt`).
