import os
import cv2
import dlib
import numpy as np
from datetime import datetime

# Initialize models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def get_face_encodings(image_path):
    """Returns face encodings for all faces in an image"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return []
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(image, 1)
    
    encodings = []
    for i, det in enumerate(dets):
        shape = sp(image, det)
        face_encoding = facerec.compute_face_descriptor(image, shape)
        encodings.append(np.array(face_encoding))
        
        # Show training faces (for verification)
        cv2.rectangle(image, (det.left(), det.top()), (det.right(), det.bottom()), (0, 255, 0), 2)
        cv2.putText(image, f"Training {os.path.basename(image_path)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Training Preview", image)
        cv2.waitKey(100)  # Brief pause to see each face
    
    return encodings

def load_known_faces(photos_dir="photos"):
    """Load and verify training data"""
    known_face_encodings = []
    known_face_names = []
    
    print("\n===== Training Started =====")
    
    for person_name in os.listdir(photos_dir):
        person_dir = os.path.join(photos_dir, person_name)
        
        if not os.path.isdir(person_dir):
            continue
            
        print(f"\nTraining on {person_name}'s photos:")
        
        for photo_file in os.listdir(person_dir):
            photo_path = os.path.join(person_dir, photo_file)
            print(f"- Processing {photo_file}")
            
            encodings = get_face_encodings(photo_path)
            if encodings:
                known_face_encodings.extend(encodings)
                known_face_names.extend([person_name] * len(encodings))
            else:
                print(f"  Warning: No faces found in {photo_file}")
    
    cv2.destroyWindow("Training Preview")
    
    if not known_face_encodings:
        print("\nERROR: No faces found in any training photos!")
        print("Please check:")
        print("1. Your photos directory structure")
        print("2. Photo quality (clear front-facing faces)")
        exit()
    
    print("\n===== Training Summary =====")
    print(f"Total people: {len(set(known_face_names))}")
    print(f"Total face samples: {len(known_face_encodings)}")
    print("===== Ready for Recognition =====\n")
    
    return known_face_encodings, known_face_names

def recognize_faces():
    known_face_encodings, known_face_names = load_known_faces()
    
    video_capture = cv2.VideoCapture(0)
    print("Starting recognition. Press 'q' to quit.")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector(rgb_frame, 1)
        
        for det in dets:
            shape = sp(rgb_frame, det)
            face_encoding = facerec.compute_face_descriptor(rgb_frame, shape)
            
            # Compare with known faces
            distances = [np.linalg.norm(known_encoding - face_encoding) 
                        for known_encoding in known_face_encodings]
            
            best_match = np.argmin(distances)
            name = "Unknown"
            
            # Only recognize if distance is below threshold
            if distances[best_match] < 0.6:  # Lower = more strict
                name = known_face_names[best_match]
                confidence = 1 - distances[best_match]
                print(f"{name} detected (Confidence: {confidence:.2f})")
            
            # Draw results
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), color, 2)
            cv2.putText(frame, f"{name}", (det.left(), det.top() - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()