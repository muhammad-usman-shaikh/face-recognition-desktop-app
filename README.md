# Face Recognition System

A Python-based face recognition system that:

- Recognizes known faces from a photos directory
- Automatically saves and tracks unknown faces
- Prevents duplicate saving of the same unknown person

## Features

- Face detection and recognition
- Automatic organization of unknown faces
- Real-time webcam recognition
- Simple training from photo directories

## Requirements

- Python 3.6+
- Webcam
- Linux/macOS/Windows

## Installation

1.  Clone the repository:

2.  python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`

3.  pip install -r requirements.txt

4.  Download Model Files

        wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

        wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
        bunzip2 \*.bz2

5.  python main.py
