import os
import cv2
import dlib
import numpy as np
from datetime import datetime
import uuid

# Initialize models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Global variables for unknown faces tracking
unknown_faces_encodings = []
unknown_faces_ids = []

def ensure_unknown_dir():
    """Ensure the unknown directory structure exists"""
    if not os.path.exists("unknown"):
        os.makedirs("unknown")

def save_unknown_face(frame, face_location, face_encoding):
    """Save an unknown face with larger portion of the image"""
    global unknown_faces_encodings, unknown_faces_ids
    
    # Check if this face matches any previously saved unknown faces
    if unknown_faces_encodings:
        distances = [np.linalg.norm(known_encoding - face_encoding) 
                    for known_encoding in unknown_faces_encodings]
        min_distance = min(distances) if distances else 1.0
        if min_distance < 0.6:  # Same person threshold
            matched_id = unknown_faces_ids[distances.index(min_distance)]
            print(f"Unknown face matches previously saved ID: {matched_id}")
            return matched_id
    
    # If new unknown face, save it with larger portion
    face_id = str(uuid.uuid4())
    face_dir = os.path.join("unknown", face_id)
    os.makedirs(face_dir, exist_ok=True)
    
    # Extract larger area around the face (from head to upper body)
    top, right, bottom, left = face_location
    height = bottom - top
    width = right - left
    
    # Expand the cropping area (adjust these multipliers as needed)
    expanded_top = max(0, top - height * 2)  # 2x height above head
    expanded_bottom = min(frame.shape[0], bottom + height * 1)  # 1x height below chin
    expanded_left = max(0, left - width * 1)  # 1x width to left
    expanded_right = min(frame.shape[1], right + width * 1)  # 1x width to right
    
    # Get the expanded image
    expanded_face_image = frame[expanded_top:expanded_bottom, expanded_left:expanded_right]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    face_path = os.path.join(face_dir, f"{timestamp}.jpg")
    cv2.imwrite(face_path, expanded_face_image)
    
    # Add to our memory of unknown faces
    unknown_faces_encodings.append(face_encoding)
    unknown_faces_ids.append(face_id)
    
    print(f"Saved new unknown face with ID: {face_id}")
    return face_id

def load_unknown_faces():
    """Load all previously saved unknown faces"""
    global unknown_faces_encodings, unknown_faces_ids
    
    if not os.path.exists("unknown"):
        return
    
    print("\nLoading previously saved unknown faces...")
    
    for face_id in os.listdir("unknown"):
        face_dir = os.path.join("unknown", face_id)
        if not os.path.isdir(face_dir):
            continue
        
        # Get the first saved image of this unknown person
        for file in os.listdir(face_dir):
            if file.endswith(".jpg"):
                image_path = os.path.join(face_dir, file)
                encodings = get_face_encodings(image_path)
                if encodings:
                    unknown_faces_encodings.extend(encodings)
                    unknown_faces_ids.extend([face_id] * len(encodings))
                    print(f"- Loaded unknown face ID: {face_id}")
                break

def get_face_encodings(image_path):
    """Returns face encodings for all faces in an image"""
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = detector(image, 1)
    
    encodings = []
    for det in dets:
        shape = sp(image, det)
        face_encoding = facerec.compute_face_descriptor(image, shape)
        encodings.append(np.array(face_encoding))
    
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
    
    if not known_face_encodings:
        print("\nERROR: No faces found in any training photos!")
        exit()
    
    print("\n===== Training Complete =====")
    return known_face_encodings, known_face_names

def recognize_faces():
    ensure_unknown_dir()
    load_unknown_faces()
    known_face_encodings, known_face_names = load_known_faces()
    
    video_capture = cv2.VideoCapture(0)
    print("\nStarting recognition. Press 'q' to quit.")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector(rgb_frame, 1)
        
        for det in dets:
            shape = sp(rgb_frame, det)
            face_encoding = facerec.compute_face_descriptor(rgb_frame, shape)
            face_encoding = np.array(face_encoding)
            
            # Compare with known faces first
            distances = [np.linalg.norm(known_encoding - face_encoding) 
                        for known_encoding in known_face_encodings]
            best_match = np.argmin(distances) if distances else -1
            name = "Unknown"
            color = (0, 0, 255)  # Red for unknown
            
            if distances and distances[best_match] < 0.6:
                name = known_face_names[best_match]
                color = (0, 255, 0)  # Green for known
            else:
                # Handle unknown face - pass the full frame and face location
                face_location = (det.top(), det.right(), det.bottom(), det.left())
                save_unknown_face(frame, face_location, face_encoding)
            
            # Draw results
            cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), color, 2)
            cv2.putText(frame, name, (det.left(), det.top() - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()