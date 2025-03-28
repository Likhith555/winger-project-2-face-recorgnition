import cv2
import os
import numpy as np
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

# Constants
DATA_FILE = 'face_data.pkl'
SAVE_FOLDER = 'dataset'
TARGET_IMAGES = 30  # Reduced for testing (original: 80-100)
MIN_FACE_SIZE = 100
CAPTURE_INTERVAL = 0.5  # Seconds between captures

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize face detector
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Create directories
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        
        # Initialize face data
        self.face_data = {'encodings': [], 'names': []}
        self._load_existing_data()
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Training variables
        self.current_person = None
        self.captured_images = 0
        self.capture_stage = "front"
        self.stage_targets = {
            "front": 10,  # Reduced for testing
            "left": 7,
            "right": 7,
            "up": 6
        }
        self.instructions = {
            "front": "Look straight ahead",
            "left": "Turn head LEFT",
            "right": "Turn head RIGHT",
            "up": "Look UP"
        }
        self.last_capture_time = 0
        self.capture_cooldown = False
        
        # Recognition variables
        self.recognition_mode = True if len(self.face_data['encodings']) > 0 else False
        self.focus_face = None
        self.last_recognition_time = 0

    def _load_existing_data(self):
        """Load existing face data if available"""
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'rb') as f:
                    self.face_data = pickle.load(f)
                    print(f"Loaded {len(self.face_data['names'])} face samples")
            
            if os.path.exists('trained_model.yml'):
                self.face_recognizer.read('trained_model.yml')
                print("Loaded trained model")
                # Verify the model is trained
                if not self.is_model_trained():
                    print("Model needs retraining")
                    self._retrain_model()
        except Exception as e:
            print(f"Error loading data: {e}")
            

    def is_model_trained(self):
        """Check if the model is properly trained"""
        try:
            # Try a dummy prediction to check if model is ready
            dummy = np.zeros((100, 100), dtype=np.uint8)
            _ = self.face_recognizer.predict(dummy)
            return True
        except:
            return False

    def _save_data(self):
        """Save all face data and model"""
        try:
            with open(DATA_FILE, 'wb') as f:
                pickle.dump(self.face_data, f)
            self.face_recognizer.write('trained_model.yml')
            print("Data and model saved successfully")
        except Exception as e:
            print(f"Error saving data: {e}")

    def detect_faces(self, frame):
        """Detect all faces in frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, 
                scaleFactor=1.2,
                minNeighbors=7,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
            return faces
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    def _start_training_new_face(self):
        """Switch to training mode for new face with user-provided name"""
        if self.current_person is None:  # Only start if not already training
            self.recognition_mode = False
            self.captured_images = 0
            self.capture_stage = "front"
            
            # Prompt user for name
            new_name = input("Enter name for the new person: ").strip()
            if not new_name:
                new_name = f"Person_{len(set(self.face_data['names'])) + 1:03d}"
            
            self.current_person = new_name
            person_folder = os.path.join(SAVE_FOLDER, self.current_person)
            os.makedirs(person_folder, exist_ok=True)
            print(f"Starting training for {self.current_person}")

    def run(self):
        print("\nFace Recognition System Started")
        print(f"Capture Rate: {1/CAPTURE_INTERVAL:.1f} faces/second")
        print("Press 'q' to quit")
        print("Press 't' to manually start training\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Detect all faces
                faces = self.detect_faces(frame)
                
                # Select focus face (largest face)
                focus_face = None
                if len(faces) > 0:
                    areas = [w * h for (x, y, w, h) in faces]
                    focus_face = faces[np.argmax(areas)]
                
                # Process all faces
                for (x, y, w, h) in faces:
                    if focus_face is not None and np.array_equal((x, y, w, h), focus_face):
                        self.process_focused_face(frame, (x, y, w, h))
                        self.focus_face = focus_face
                    else:
                        # Blur other faces
                        frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (101, 101), 50)
                
                # Display mode info
                mode_text = "Training" if not self.recognition_mode else "Recognition"
                cv2.putText(frame, f"Mode: {mode_text}", (20, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Face Recognition', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self._start_training_new_face()
                
        except Exception as e:
            print(f"Critical error: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown()
            self._save_data()
            print("System shutdown complete")   

    def _recognize_face(self, face_gray, frame, x, y):
        """Recognize a face and display results"""
        try:
            if not self.is_model_trained():
                cv2.putText(frame, "Model not trained", (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return
            
            label, confidence = self.face_recognizer.predict(face_gray)
            
            if confidence < 70 and label < len(self.face_data['names']):
                # Known face
                name = self.face_data['names'][label]
                cv2.putText(frame, f"{name} ({confidence:.1f})", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Unknown face
                if time.time() - self.last_recognition_time > 3:
                    self._start_training_new_face()
                cv2.putText(frame, "Unknown", (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            print(f"Recognition error: {e}")
            cv2.putText(frame, "Error", (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _start_training_new_face(self):
        """Switch to training mode for new face"""
        if self.current_person is None:  # Only start if not already training
            self.recognition_mode = False
            self.captured_images = 0
            self.capture_stage = "front"
            self.current_person = f"Person_{len(set(self.face_data['names'])) + 1:03d}"
            person_folder = os.path.join(SAVE_FOLDER, self.current_person)
            os.makedirs(person_folder, exist_ok=True)
            print(f"Starting training for {self.current_person}")

    def process_focused_face(self, frame, face):
        """Process the focused face with controlled capture rate"""
        x, y, w, h = face
        current_time = time.time()
        
        # Draw visual feedback
        border_color = (0, 200, 0) if not self.capture_cooldown else (0, 120, 120)
        cv2.rectangle(frame, (x, y), (x+w, y+h), border_color, 2)
        
        if not self.recognition_mode and self.current_person is not None:
            # Training mode with controlled capture
            if current_time - self.last_capture_time >= CAPTURE_INTERVAL:
                try:
                    face_roi = frame[y:y+h, x:x+w]
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    self._train_face(face_gray, frame, x, y, w, h)
                    self.last_capture_time = current_time
                    self.capture_cooldown = True
                    cv2.waitKey(1)  # Allow GUI update
                    self.capture_cooldown = False
                except Exception as e:
                    print(f"Face processing error: {e}")
            else:
                # Display countdown
                remaining = CAPTURE_INTERVAL - (current_time - self.last_capture_time)
                cv2.putText(frame, f"Next: {remaining:.1f}s", 
                          (x, y-90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            # Recognition mode
            try:
                face_roi = frame[y:y+h, x:x+w]
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                self._recognize_face(face_gray, frame, x, y)
                self.last_recognition_time = time.time()
            except Exception as e:
                print(f"Recognition processing error: {e}")
        
        return frame

    def _train_face(self, face_gray, frame, x, y, w, h):
        """Handle face training with controlled capture rate"""
        try:
            if self.current_person is None:
                raise ValueError("No current person set for training")
                
            # Create person folder path safely
            person_folder = os.path.join(SAVE_FOLDER, self.current_person)
            if not os.path.exists(person_folder):
                os.makedirs(person_folder, exist_ok=True)
            
            # Save face image
            img_path = os.path.join(person_folder, f"{self.capture_stage}_{self.captured_images:03d}.jpg")
            cv2.imwrite(img_path, face_gray)
            
            # Add to training data
            self.face_data['encodings'].append(face_gray)
            self.face_data['names'].append(self.current_person)
            self.captured_images += 1
            
            # Display instructions
            cv2.putText(frame, self.instructions[self.capture_stage], 
                      (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Captured: {self.captured_images}/{TARGET_IMAGES}", 
                      (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Visual feedback
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            time.sleep(0.05)
            
            # Check stage completion
            if self.captured_images >= self.stage_targets.get(self.capture_stage, 0):
                self._advance_training_stage()
        except Exception as e:
            print(f"Training error: {e}")

    def _advance_training_stage(self):
        """Move to next training stage or complete training"""
        try:
            stages = list(self.stage_targets.keys())
            current_idx = stages.index(self.capture_stage)
            
            if current_idx < len(stages) - 1:
                self.capture_stage = stages[current_idx + 1]
                print(f"Please {self.instructions[self.capture_stage]}")
            elif self.captured_images >= TARGET_IMAGES:
                print(f"Training complete for {self.current_person}!")
                self._retrain_model()
                self.recognition_mode = True
                self.current_person = None
        except Exception as e:
            print(f"Stage advancement error: {e}")

    def _retrain_model(self):
        """Retrain the recognition model"""
        try:
            if len(self.face_data['encodings']) > 0:
                labels = [i for i, _ in enumerate(self.face_data['names'])]
                self.face_recognizer.train(self.face_data['encodings'], np.array(labels))
                self._save_data()
                print("Model retrained successfully")
        except Exception as e:
            print(f"Training error: {e}")

    def run(self):
        print("\nFace Recognition System Started")
        print(f"Capture Rate: {1/CAPTURE_INTERVAL:.1f} faces/second")
        print("Press 'q' to quit")
        print("Press 't' to manually start training\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Detect all faces
                faces = self.detect_faces(frame)
                
                # Select focus face (largest face)
                focus_face = None
                if len(faces) > 0:
                    areas = [w * h for (x, y, w, h) in faces]
                    focus_face = faces[np.argmax(areas)]
                
                # Process all faces
                for (x, y, w, h) in faces:
                    if focus_face is not None and np.array_equal((x, y, w, h), focus_face):
                        self.process_focused_face(frame, (x, y, w, h))
                        self.focus_face = focus_face
                    else:
                        # Blur other faces
                        frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (101, 101), 50)
                
                # Display mode info
                mode_text = "Training" if not self.recognition_mode else "Recognition"
                cv2.putText(frame, f"Mode: {mode_text}", (20, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Face Recognition', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self._start_training_new_face()
                
        except Exception as e:
            print(f"Critical error: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown()
            self._save_data()
            print("System shutdown complete")
            

if __name__ == "__main__":
    try:
        system = FaceRecognitionSystem()
        system.run()
    except Exception as e:
        print(f"Fatal error: {e}")