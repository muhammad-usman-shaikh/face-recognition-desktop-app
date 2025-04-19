# import os
# import cv2
# import dlib
# import numpy as np
# from datetime import datetime
# import uuid

# # Initialize models
# detector = dlib.get_frontal_face_detector()
# sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# # Global variables for unknown faces tracking
# unknown_faces_encodings = []
# unknown_faces_ids = []

# def ensure_unknown_dir():
#     """Ensure the unknown directory structure exists"""
#     if not os.path.exists("unknown"):
#         os.makedirs("unknown")

# def save_unknown_face(frame, face_location, face_encoding):
#     """Save an unknown face with larger portion of the image"""
#     global unknown_faces_encodings, unknown_faces_ids
    
#     # Check if this face matches any previously saved unknown faces
#     if unknown_faces_encodings:
#         distances = [np.linalg.norm(known_encoding - face_encoding) 
#                     for known_encoding in unknown_faces_encodings]
#         min_distance = min(distances) if distances else 1.0
#         if min_distance < 0.6:  # Same person threshold
#             matched_id = unknown_faces_ids[distances.index(min_distance)]
#             print(f"Unknown face matches previously saved ID: {matched_id}")
#             return matched_id
    
#     # If new unknown face, save it with larger portion
#     face_id = str(uuid.uuid4())
#     face_dir = os.path.join("unknown", face_id)
#     os.makedirs(face_dir, exist_ok=True)
    
#     # Extract larger area around the face (from head to upper body)
#     top, right, bottom, left = face_location
#     height = bottom - top
#     width = right - left
    
#     # Expand the cropping area (adjust these multipliers as needed)
#     expanded_top = max(0, top - height * 2)  # 2x height above head
#     expanded_bottom = min(frame.shape[0], bottom + height * 1)  # 1x height below chin
#     expanded_left = max(0, left - width * 1)  # 1x width to left
#     expanded_right = min(frame.shape[1], right + width * 1)  # 1x width to right
    
#     # Get the expanded image
#     expanded_face_image = frame[expanded_top:expanded_bottom, expanded_left:expanded_right]
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     face_path = os.path.join(face_dir, f"{timestamp}.jpg")
#     cv2.imwrite(face_path, expanded_face_image)
    
#     # Add to our memory of unknown faces
#     unknown_faces_encodings.append(face_encoding)
#     unknown_faces_ids.append(face_id)
    
#     print(f"Saved new unknown face with ID: {face_id}")
#     return face_id

# def load_unknown_faces():
#     """Load all previously saved unknown faces"""
#     global unknown_faces_encodings, unknown_faces_ids
    
#     if not os.path.exists("unknown"):
#         return
    
#     print("\nLoading previously saved unknown faces...")
    
#     for face_id in os.listdir("unknown"):
#         face_dir = os.path.join("unknown", face_id)
#         if not os.path.isdir(face_dir):
#             continue
        
#         # Get the first saved image of this unknown person
#         for file in os.listdir(face_dir):
#             if file.endswith(".jpg"):
#                 image_path = os.path.join(face_dir, file)
#                 encodings = get_face_encodings(image_path)
#                 if encodings:
#                     unknown_faces_encodings.extend(encodings)
#                     unknown_faces_ids.extend([face_id] * len(encodings))
#                     print(f"- Loaded unknown face ID: {face_id}")
#                 break

# def get_face_encodings(image_path):
#     """Returns face encodings for all faces in an image"""
#     image = cv2.imread(image_path)
#     if image is None:
#         return []
    
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     dets = detector(image, 1)
    
#     encodings = []
#     for det in dets:
#         shape = sp(image, det)
#         face_encoding = facerec.compute_face_descriptor(image, shape)
#         encodings.append(np.array(face_encoding))
    
#     return encodings

# def load_known_faces(photos_dir="photos"):
#     """Load and verify training data"""
#     known_face_encodings = []
#     known_face_names = []
    
#     print("\n===== Training Started =====")
    
#     for person_name in os.listdir(photos_dir):
#         person_dir = os.path.join(photos_dir, person_name)
        
#         if not os.path.isdir(person_dir):
#             continue
            
#         print(f"\nTraining on {person_name}'s photos:")
        
#         for photo_file in os.listdir(person_dir):
#             photo_path = os.path.join(person_dir, photo_file)
#             print(f"- Processing {photo_file}")
            
#             encodings = get_face_encodings(photo_path)
#             if encodings:
#                 known_face_encodings.extend(encodings)
#                 known_face_names.extend([person_name] * len(encodings))
#             else:
#                 print(f"  Warning: No faces found in {photo_file}")
    
#     if not known_face_encodings:
#         print("\nERROR: No faces found in any training photos!")
#         exit()
    
#     print("\n===== Training Complete =====")
#     return known_face_encodings, known_face_names

# def recognize_faces():
#     ensure_unknown_dir()
#     load_unknown_faces()
#     known_face_encodings, known_face_names = load_known_faces()
    
#     video_capture = cv2.VideoCapture(0)
#     print("\nStarting recognition. Press 'q' to quit.")
    
#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             continue
        
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         dets = detector(rgb_frame, 1)
        
#         for det in dets:
#             shape = sp(rgb_frame, det)
#             face_encoding = facerec.compute_face_descriptor(rgb_frame, shape)
#             face_encoding = np.array(face_encoding)
            
#             # Compare with known faces first
#             distances = [np.linalg.norm(known_encoding - face_encoding) 
#                         for known_encoding in known_face_encodings]
#             best_match = np.argmin(distances) if distances else -1
#             name = "Unknown"
#             color = (0, 0, 255)  # Red for unknown
            
#             if distances and distances[best_match] < 0.6:
#                 name = known_face_names[best_match]
#                 color = (0, 255, 0)  # Green for known
#             else:
#                 # Handle unknown face - pass the full frame and face location
#                 face_location = (det.top(), det.right(), det.bottom(), det.left())
#                 save_unknown_face(frame, face_location, face_encoding)
            
#             # Draw results
#             cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), color, 2)
#             cv2.putText(frame, name, (det.left(), det.top() - 10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
#         cv2.imshow('Face Recognition', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     video_capture.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     recognize_faces()

# V2
# import os
# import cv2
# import dlib
# import numpy as np
# from datetime import datetime
# import uuid
# import shutil

# # Initialize models
# detector = dlib.get_frontal_face_detector()
# sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# # Global variables for unknown faces tracking
# unknown_faces_encodings = []
# unknown_faces_ids = []
# unknown_faces_info = {}  # Stores info about unknown faces (path, encodings)

# # Thresholds
# KNOWN_FACE_THRESHOLD = 0.5  # Lower is more strict
# UNKNOWN_FACE_MATCH_THRESHOLD = 0.6  # For matching unknown faces to each other

# def ensure_dirs():
#     """Ensure required directory structure exists"""
#     if not os.path.exists("unknown"):
#         os.makedirs("unknown")
#     if not os.path.exists("photos"):
#         os.makedirs("photos")

# def save_unknown_face(frame, face_location, face_encoding):
#     """Save an unknown face with larger portion of the image"""
#     global unknown_faces_encodings, unknown_faces_ids, unknown_faces_info
    
#     # Check if this face matches any previously saved unknown faces
#     if unknown_faces_encodings:
#         distances = [np.linalg.norm(known_encoding - face_encoding) 
#                     for known_encoding in unknown_faces_encodings]
#         min_distance = min(distances) if distances else 1.0
#         if min_distance < UNKNOWN_FACE_MATCH_THRESHOLD:  # Same person threshold
#             matched_id = unknown_faces_ids[distances.index(min_distance)]
#             print(f"Unknown face matches previously saved ID: {matched_id}")
#             return matched_id
    
#     # If new unknown face, save it with larger portion
#     face_id = str(uuid.uuid4())
#     face_dir = os.path.join("unknown", face_id)
#     os.makedirs(face_dir, exist_ok=True)
    
#     # Extract larger area around the face (from head to upper body)
#     top, right, bottom, left = face_location
#     height = bottom - top
#     width = right - left
    
#     # Expand the cropping area (adjust these multipliers as needed)
#     expanded_top = max(0, top - height * 2)  # 2x height above head
#     expanded_bottom = min(frame.shape[0], bottom + height * 1)  # 1x height below chin
#     expanded_left = max(0, left - width * 1)  # 1x width to left
#     expanded_right = min(frame.shape[1], right + width * 1)  # 1x width to right
    
#     # Get the expanded image
#     expanded_face_image = frame[expanded_top:expanded_bottom, expanded_left:expanded_right]
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     face_path = os.path.join(face_dir, f"{timestamp}.jpg")
#     cv2.imwrite(face_path, expanded_face_image)
    
#     # Add to our memory of unknown faces
#     unknown_faces_encodings.append(face_encoding)
#     unknown_faces_ids.append(face_id)
#     unknown_faces_info[face_id] = {
#         'path': face_dir,
#         'encodings': [face_encoding],
#         'images': [face_path]
#     }
    
#     print(f"Saved new unknown face with ID: {face_id}")
#     return face_id

# def load_unknown_faces():
#     """Load all previously saved unknown faces"""
#     global unknown_faces_encodings, unknown_faces_ids, unknown_faces_info
    
#     if not os.path.exists("unknown"):
#         return
    
#     print("\nLoading previously saved unknown faces...")
    
#     for face_id in os.listdir("unknown"):
#         face_dir = os.path.join("unknown", face_id)
#         if not os.path.isdir(face_dir):
#             continue
        
#         # Initialize info for this unknown face
#         unknown_faces_info[face_id] = {
#             'path': face_dir,
#             'encodings': [],
#             'images': []
#         }
        
#         # Process all images for this unknown person
#         for file in os.listdir(face_dir):
#             if file.endswith(".jpg"):
#                 image_path = os.path.join(face_dir, file)
#                 encodings = get_face_encodings(image_path)
#                 if encodings:
#                     unknown_faces_encodings.extend(encodings)
#                     unknown_faces_ids.extend([face_id] * len(encodings))
#                     unknown_faces_info[face_id]['encodings'].extend(encodings)
#                     unknown_faces_info[face_id]['images'].append(image_path)
        
#         print(f"- Loaded unknown face ID: {face_id} with {len(unknown_faces_info[face_id]['images'])} images")

# def get_face_encodings(image_path):
#     """Returns face encodings for all faces in an image"""
#     image = cv2.imread(image_path)
#     if image is None:
#         return []
    
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     dets = detector(image, 1)
    
#     encodings = []
#     for det in dets:
#         shape = sp(image, det)
#         face_encoding = facerec.compute_face_descriptor(image, shape)
#         encodings.append(np.array(face_encoding))
    
#     return encodings

# def load_known_faces(photos_dir="photos"):
#     """Load and verify training data"""
#     known_face_encodings = []
#     known_face_names = []
    
#     print("\n===== Training Started =====")
    
#     for person_name in os.listdir(photos_dir):
#         person_dir = os.path.join(photos_dir, person_name)
        
#         if not os.path.isdir(person_dir):
#             continue
            
#         print(f"\nTraining on {person_name}'s photos:")
        
#         for photo_file in os.listdir(person_dir):
#             photo_path = os.path.join(person_dir, photo_file)
#             print(f"- Processing {photo_file}")
            
#             encodings = get_face_encodings(photo_path)
#             if encodings:
#                 known_face_encodings.extend(encodings)
#                 known_face_names.extend([person_name] * len(encodings))
#             else:
#                 print(f"  Warning: No faces found in {photo_file}")
    
#     if not known_face_encodings:
#         print("\nWARNING: No faces found in any training photos!")
    
#     print("\n===== Training Complete =====")
#     return known_face_encodings, known_face_names

# def identify_unknown_faces():
#     """Show unknown faces and allow naming them"""
#     global unknown_faces_encodings, unknown_faces_ids, unknown_faces_info
    
#     if not unknown_faces_info:
#         print("No unknown faces to identify!")
#         return
    
#     print("\n===== Identifying Unknown Faces =====")
    
#     for face_id, info in unknown_faces_info.items():
#         # Display the first image of this unknown person
#         if not info['images']:
#             continue
            
#         image = cv2.imread(info['images'][0])
#         cv2.imshow("Unknown Face - Press 'n' to name, 's' to skip", image)
        
#         key = cv2.waitKey(0) & 0xFF
        
#         if key == ord('n'):  # Name this face
#             name = input(f"Enter name for face {face_id}: ").strip()
#             if name:
#                 # Create directory for this person if it doesn't exist
#                 person_dir = os.path.join("photos", name)
#                 os.makedirs(person_dir, exist_ok=True)
                
#                 # Move all images of this unknown face to the known directory
#                 for i, img_path in enumerate(info['images']):
#                     new_path = os.path.join(person_dir, f"{face_id}_{i}.jpg")
#                     shutil.move(img_path, new_path)
                
#                 print(f"Moved {len(info['images'])} images to {person_dir}")
                
#                 # Remove this face from unknown tracking
#                 del unknown_faces_info[face_id]
#                 # Remove all encodings for this face
#                 indices = [i for i, id_val in enumerate(unknown_faces_ids) if id_val == face_id]
#                 for index in sorted(indices, reverse=True):
#                     del unknown_faces_encodings[index]
#                     del unknown_faces_ids[index]
                
#         elif key == ord('s'):  # Skip
#             continue
            
#         cv2.destroyAllWindows()
    
#     print("\n===== Identification Complete =====")

# def recognize_faces():
#     ensure_dirs()
#     load_unknown_faces()
#     known_face_encodings, known_face_names = load_known_faces()
    
#     video_capture = cv2.VideoCapture(0)
#     print("\nStarting recognition. Press 'q' to quit, 'i' to identify unknowns, 't' to retrain.")
    
#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             continue
        
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         dets = detector(rgb_frame, 1)
        
#         for det in dets:
#             shape = sp(rgb_frame, det)
#             face_encoding = facerec.compute_face_descriptor(rgb_frame, shape)
#             face_encoding = np.array(face_encoding)
            
#             # Compare with known faces first
#             distances = []
#             if known_face_encodings:
#                 distances = [np.linalg.norm(known_encoding - face_encoding) 
#                             for known_encoding in known_face_encodings]
#                 best_match = np.argmin(distances) if distances else -1
#             else:
#                 best_match = -1
                
#             name = "Unknown"
#             color = (0, 0, 255)  # Red for unknown
            
#             if distances and distances[best_match] < KNOWN_FACE_THRESHOLD:
#                 name = known_face_names[best_match]
#                 color = (0, 255, 0)  # Green for known
#             else:
#                 # Handle unknown face - pass the full frame and face location
#                 face_location = (det.top(), det.right(), det.bottom(), det.left())
#                 save_unknown_face(frame, face_location, face_encoding)
            
#             # Draw results
#             cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), color, 2)
#             cv2.putText(frame, name, (det.left(), det.top() - 10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
#         # Add buttons/instructions to the frame
#         cv2.putText(frame, "Press 'i' to Identify Unknowns", (10, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(frame, "Press 't' to Retrain", (10, 60), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
#         cv2.imshow('Face Recognition', frame)
        
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('i'):
#             identify_unknown_faces()
#             # Refresh the display
#             cv2.imshow('Face Recognition', frame)
#         elif key == ord('t'):
#             # Reload known faces
#             known_face_encodings, known_face_names = load_known_faces()
#             print("Model retrained with current known faces!")
    
#     video_capture.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     recognize_faces()


# V3
import os
import cv2
import dlib
import numpy as np
from datetime import datetime
import uuid
import shutil
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# Initialize models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Global variables
known_face_encodings = []
known_face_names = []
unknown_faces_info = {}
unknown_faces_encodings = []
unknown_faces_ids = []
current_face_index = 0
identify_window = None

# Thresholds
KNOWN_FACE_THRESHOLD = 0.5
UNKNOWN_FACE_MATCH_THRESHOLD = 0.6

def ensure_dirs():
    """Ensure required directory structure exists"""
    if not os.path.exists("unknown"):
        os.makedirs("unknown")
    if not os.path.exists("photos"):
        os.makedirs("photos")

def save_unknown_face(frame, face_location, face_encoding):
    """Save an unknown face with larger portion of the image"""
    global unknown_faces_encodings, unknown_faces_ids, unknown_faces_info
    
    # Check if this face matches any previously saved unknown faces
    if unknown_faces_encodings:
        distances = [np.linalg.norm(known_encoding - face_encoding) 
                    for known_encoding in unknown_faces_encodings]
        min_distance = min(distances) if distances else 1.0
        if min_distance < UNKNOWN_FACE_MATCH_THRESHOLD:  # Same person threshold
            matched_id = unknown_faces_ids[distances.index(min_distance)]
            print(f"Unknown face matches previously saved ID: {matched_id}")
            return matched_id
    
    # If new unknown face, save it with larger portion
    face_id = str(uuid.uuid4())
    face_dir = os.path.join("unknown", face_id)
    os.makedirs(face_dir, exist_ok=True)
    
    # Extract larger area around the face
    top, right, bottom, left = face_location
    height = bottom - top
    width = right - left
    
    # Expand the cropping area
    expanded_top = max(0, top - height * 2)
    expanded_bottom = min(frame.shape[0], bottom + height * 1)
    expanded_left = max(0, left - width * 1)
    expanded_right = min(frame.shape[1], right + width * 1)
    
    # Get the expanded image
    expanded_face_image = frame[expanded_top:expanded_bottom, expanded_left:expanded_right]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    face_path = os.path.join(face_dir, f"{timestamp}.jpg")
    cv2.imwrite(face_path, expanded_face_image)
    
    # Add to our memory of unknown faces
    unknown_faces_encodings.append(face_encoding)
    unknown_faces_ids.append(face_id)
    unknown_faces_info[face_id] = {
        'path': face_dir,
        'encodings': [face_encoding],
        'images': [face_path]
    }
    
    print(f"Saved new unknown face with ID: {face_id}")
    return face_id

def load_unknown_faces():
    """Load all previously saved unknown faces"""
    global unknown_faces_encodings, unknown_faces_ids, unknown_faces_info
    
    if not os.path.exists("unknown"):
        return
    
    print("\nLoading previously saved unknown faces...")
    
    for face_id in os.listdir("unknown"):
        face_dir = os.path.join("unknown", face_id)
        if not os.path.isdir(face_dir):
            continue
        
        # Initialize info for this unknown face
        unknown_faces_info[face_id] = {
            'path': face_dir,
            'encodings': [],
            'images': []
        }
        
        # Process all images for this unknown person
        for file in os.listdir(face_dir):
            if file.endswith(".jpg"):
                image_path = os.path.join(face_dir, file)
                encodings = get_face_encodings(image_path)
                if encodings:
                    unknown_faces_encodings.extend(encodings)
                    unknown_faces_ids.extend([face_id] * len(encodings))
                    unknown_faces_info[face_id]['encodings'].extend(encodings)
                    unknown_faces_info[face_id]['images'].append(image_path)
        
        print(f"- Loaded unknown face ID: {face_id} with {len(unknown_faces_info[face_id]['images'])} images")

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
    global known_face_encodings, known_face_names
    
    known_face_encodings = []
    known_face_names = []
    
    print("\n===== Training Started =====")
    
    if not os.path.exists(photos_dir):
        print("No photos directory found!")
        return [], []
    
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
        print("\nWARNING: No faces found in any training photos!")
    
    print("\n===== Training Complete =====")
    return known_face_encodings, known_face_names

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.video_capture = cv2.VideoCapture(0)
        self.setup_ui()
        self.load_data()
        self.update()

    def setup_ui(self):
        """Setup the main application window"""
        self.master.title("Face Recognition System")
        self.master.geometry("800x600")
        
        # Video display frame
        self.video_frame = tk.Label(self.master)
        self.video_frame.pack(expand=True, fill=tk.BOTH)
        
        # Control buttons frame
        control_frame = tk.Frame(self.master)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.identify_btn = tk.Button(control_frame, text="Identify Unknown Faces", 
                                    command=self.show_identify_window)
        self.identify_btn.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = tk.Button(control_frame, text="Retrain Model", 
                                 command=self.retrain_model)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.quit_btn = tk.Button(control_frame, text="Quit", 
                                command=self.quit_app)
        self.quit_btn.pack(side=tk.RIGHT, padx=5)

    def load_data(self):
        """Load known and unknown faces"""
        ensure_dirs()
        load_unknown_faces()
        load_known_faces()

    def update(self):
        """Update the video feed"""
        ret, frame = self.video_capture.read()
        if ret:
            frame = self.process_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
        self.master.after(10, self.update)

    def process_frame(self, frame):
        """Process each video frame for face recognition"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector(rgb_frame, 1)
        
        for det in dets:
            shape = sp(rgb_frame, det)
            face_encoding = facerec.compute_face_descriptor(rgb_frame, shape)
            face_encoding = np.array(face_encoding)
            
            # Compare with known faces
            distances = []
            if known_face_encodings:
                distances = [np.linalg.norm(known_encoding - face_encoding) 
                          for known_encoding in known_face_encodings]
                best_match = np.argmin(distances) if distances else -1
            else:
                best_match = -1
                
            name = "Unknown"
            color = (0, 0, 255)  # Red for unknown
            
            if distances and distances[best_match] < KNOWN_FACE_THRESHOLD:
                name = known_face_names[best_match]
                color = (0, 255, 0)  # Green for known
            else:
                face_location = (det.top(), det.right(), det.bottom(), det.left())
                save_unknown_face(frame, face_location, face_encoding)
            
            # Draw results
            cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), color, 2)
            cv2.putText(frame, name, (det.left(), det.top() - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return frame

    def show_identify_window(self):
        """Show window for identifying unknown faces"""
        global identify_window, current_face_index
        
        if not unknown_faces_info:
            messagebox.showinfo("Info", "No unknown faces to identify!")
            return
        
        if identify_window is not None:
            identify_window.destroy()
        
        identify_window = tk.Toplevel(self.master)
        identify_window.title("Identify Unknown Faces")
        identify_window.geometry("1000x800")
        
        current_face_index = 0
        self.show_current_face(identify_window)

    def show_current_face(self, window):
        """Show the current unknown face with controls"""
        global current_face_index
        
        face_ids = list(unknown_faces_info.keys())
        if current_face_index >= len(face_ids):
            window.destroy()
            return
        
        face_id = face_ids[current_face_index]
        info = unknown_faces_info[face_id]
        
        if not info['images']:
            current_face_index += 1
            self.show_current_face(window)
            return
        
        # Clear previous widgets
        for widget in window.winfo_children():
            widget.destroy()
        
        # Display the image
        img_frame = tk.Frame(window)
        img_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        image_path = info['images'][0]
        img = Image.open(image_path)
        
        # Calculate display size while maintaining aspect ratio
        max_width = 900
        max_height = 600
        img_width, img_height = img.size
        ratio = min(max_width/img_width, max_height/img_height)
        new_size = (int(img_width*ratio), int(img_height*ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(img_frame, image=photo)
        label.image = photo  # Keep reference
        label.pack(expand=True)
        
        # Control buttons
        btn_frame = tk.Frame(window)
        btn_frame.pack(fill=tk.X, padx=20, pady=10)
        
        name_btn = tk.Button(btn_frame, text="Name This Person", 
                           command=lambda: self.name_current_face(window, face_id))
        name_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        skip_btn = tk.Button(btn_frame, text="Skip", 
                           command=lambda: self.next_face(window))
        skip_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        delete_btn = tk.Button(btn_frame, text="Delete", 
                             command=lambda: self.delete_current_face(window, face_id))
        delete_btn.pack(side=tk.LEFT, padx=5, expand=True)
        
        quit_btn = tk.Button(btn_frame, text="Quit", 
                           command=window.destroy)
        quit_btn.pack(side=tk.RIGHT, padx=5, expand=True)

    def name_current_face(self, window, face_id):
        """Name the current unknown face"""
        name = simpledialog.askstring("Name Face", "Enter name for this person:")
        if name:
            info = unknown_faces_info[face_id]
            
            # Create directory for this person
            person_dir = os.path.join("photos", name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Move all images of this unknown face
            for i, img_path in enumerate(info['images']):
                new_path = os.path.join(person_dir, f"{face_id}_{i}.jpg")
                shutil.move(img_path, new_path)
            
            # Remove from unknown tracking
            self.remove_unknown_face(face_id)
            
            # Move to next face
            self.next_face(window)

    def delete_current_face(self, window, face_id):
        """Delete the current unknown face"""
        if messagebox.askyesno("Confirm", "Delete this face permanently?"):
            info = unknown_faces_info[face_id]
            
            # Remove all images
            for img_path in info['images']:
                if os.path.exists(img_path):
                    os.remove(img_path)
            
            # Remove directory
            if os.path.exists(info['path']):
                shutil.rmtree(info['path'])
            
            # Remove from tracking
            self.remove_unknown_face(face_id)
            
            # Move to next face
            self.next_face(window)

    def remove_unknown_face(self, face_id):
        """Remove face from unknown tracking"""
        global unknown_faces_encodings, unknown_faces_ids, unknown_faces_info
        
        # Remove from info
        if face_id in unknown_faces_info:
            del unknown_faces_info[face_id]
        
        # Remove all encodings for this face
        indices = [i for i, id_val in enumerate(unknown_faces_ids) if id_val == face_id]
        for index in sorted(indices, reverse=True):
            del unknown_faces_encodings[index]
            del unknown_faces_ids[index]

    def next_face(self, window):
        """Move to the next unknown face"""
        global current_face_index
        current_face_index += 1
        self.show_current_face(window)

    def retrain_model(self):
        """Reload known faces"""
        global known_face_encodings, known_face_names
        known_face_encodings, known_face_names = load_known_faces()
        messagebox.showinfo("Info", "Model retrained with current known faces!")

    def quit_app(self):
        """Clean up and quit application"""
        self.video_capture.release()
        self.master.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()