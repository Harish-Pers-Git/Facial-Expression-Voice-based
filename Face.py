import cv2
import numpy as np
import torch
import time
import threading
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from collections import deque
import os
import pyttsx3
import subprocess

class ThreadedVideoCapture:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.005)  # Small sleep to reduce CPU usage

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            else:
                return False, None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

class OptimizedFacialExpressionDetector:
    def __init__(self):
        """
        Initialize emotion model, face detectors, and camera with optimizations
        """
        print("Loading Hugging Face model using processor + model pattern...")
        model_name = "dima806/facial_emotions_image_detection"

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.labels = self.model.config.id2label
            print("✅ Emotion recognition model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

        # Initialize Haar cascades
        haarcascades_dir = os.path.join(os.path.dirname(cv2.__file__), "data")
        face_cascade_path = os.path.join(haarcascades_dir, 'haarcascade_frontalface_default.xml')
        smile_cascade_path = os.path.join(haarcascades_dir, 'haarcascade_smile.xml')
        if not os.path.exists(face_cascade_path) or not os.path.exists(smile_cascade_path):
            print("❌ Haarcascade files not found. Please check your OpenCV installation.")
            exit(1)
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

        # Initialize webcam with optimized settings
        # Use threaded video capture
        self.cap = ThreadedVideoCapture(0, 640, 480)
        if not self.cap.ret:
            raise ValueError("❌ Camera not accessible.")
        
        # Performance optimization variables
        self.last_emotion_time = 0
        self.emotion_cooldown = 1.5  # Process emotion every 1.5 seconds
        self.emotion_results = {}  # Cache emotion results by face position
        self.detection_history = deque(maxlen=5)
        self.last_smile_time = 0
        self.smile_cooldown = 2.0
        self.last_face_detection_time = 0
        self.face_detection_cooldown = 0.3  # Limit face detection frequency
        
        # Threading for non-blocking operations
        self.emotion_thread = None
        self.emotion_queue = {}
        self.lock = threading.Lock()
        
        # FPS tracking
        self.fps = 0.0
        
        # Face tracking for smoothing
        self.face_tracker = {}  # key: face_key, value: {'rect': (x, y, w, h), 'last_seen': frame_count, 'avg_rect': [x, y, w, h]}
        self.max_missing_frames = 5  # Number of frames to keep a face after last seen
        self.smooth_alpha = 0.4  # Smoothing factor for running average

        # Emotion message tracking
        self.last_message_time = 0
        self.message_cooldown = 3.0  # Show message every 3 seconds
        self.current_message = ""
        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
        self.tts_lock = threading.Lock()
        self.tts_thread = None
        self.sad_start_time = None
        self.sad_announced = False
        self.last_streak_emotion = None
        self.streak_start_time = None
        self.streak_announced = False
        self.last_spoken_emotion = None
        self.last_spoken_time = 0
        self.speak_cooldown = 5.0  # seconds before the same message can be spoken again
        print("📷 Camera ready. Press 'q' to quit.")

    def get_encouraging_message(self, emotion):
        """
        Return encouraging message based on detected emotion
        """
        emotion_lower = emotion.lower()
        
        if emotion_lower in ['sad', 'sadness', 'depressed']:
            return "Don't be sad, everything will be fine! 😊"
        elif emotion_lower in ['angry', 'anger', 'furious', 'mad']:
            return "Be happy, don't get stressed! 😌"
        elif emotion_lower in ['fear', 'afraid', 'scared', 'anxious']:
            return "Don't worry, you're safe and everything will be okay! 💪"
        elif emotion_lower in ['disgust', 'disgusted']:
            return "Let's focus on the positive things! 🌟"
        elif emotion_lower in ['surprise', 'surprised']:
            return "Life is full of wonderful surprises! ✨"
        elif emotion_lower in ['happy', 'joy', 'smile', 'laugh']:
            return "Keep that beautiful smile! 😄"
        elif emotion_lower in ['neutral']:
            return "Stay positive and keep going! 💫"
        else:
            return "You're doing great! Keep your head up! 🌈"

    def preprocess_face(self, face_bgr):
        """
        Convert OpenCV BGR face to RGB PIL and prepare model input
        """
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        return pil_image

    def predict_emotion_async(self, face_bgr, face_key):
        """
        Predict emotion using Hugging Face model (threaded)
        """
        try:
            pil_face = self.preprocess_face(face_bgr)
            inputs = self.processor(images=pil_face, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_class_id = int(torch.argmax(outputs.logits, dim=1))
                emotion = self.labels[predicted_class_id]

            with self.lock:
                self.emotion_results[face_key] = emotion
                print(f"✅ Emotion detected: {emotion} for face {face_key}")
        except Exception as e:
            print(f"⚠️ Error processing emotion: {e}")
            with self.lock:
                self.emotion_results[face_key] = "Error"

    def detect_faces_optimized(self, frame):
        """
        Detect faces using Haar cascade with optimized parameters
        """
        # Downsample for faster detection
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            small_frame = cv2.resize(frame, (640, int(height * scale)))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6,  # Increased for more stable detection
            minSize=(40, 40)  # Increased minimum size for more stable detection
        )

        # Scale back if we downsampled
        if width > 640:
            faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for x, y, w, h in faces]

        return faces

    def detect_smile_basic(self, face_bgr):
        """
        Fallback smile detection using Haar smile cascade
        """
        gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        smiles = self.smile_cascade.detectMultiScale(gray_face, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
        return len(smiles) > 0

    def get_face_key(self, x, y, w, h):
        """
        Create a unique key for face tracking with more tolerance
        """
        # More tolerant face tracking to avoid frequent re-processing
        return f"{x//150}_{y//150}_{w//150}_{h//150}"

    def speak_message(self, message):
        """
        Speak the encouraging message using Windows SAPI via PowerShell for reliability.
        """
        subprocess.Popen(
            ['powershell', '-Command', f"Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{message}')"]
        )

    def run_detection(self):
        """
        Main loop to perform emotion detection and display annotated video
        """
        frame_count = 0
        fps_start_time = time.time()
        fps_counter = 0

        last_face_roi = None  # For debugging
        last_face_coords = None
        last_emotion = None
        last_emotion_time = 0
        emotion_cooldown = 1.0  # seconds
        emotion_thread = None
        emotion_result = None

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None or not isinstance(frame, np.ndarray):
                print("⚠️ Failed to capture frame.")
                continue

            frame_count += 1
            current_time = time.time()

            # Detect faces every frame (for largest face selection)
            faces = self.detect_faces_optimized(frame)

            # Select the largest face
            largest_face = None
            max_area = 0
            for (x, y, w, h) in faces:
                area = w * h
                if area > max_area:
                    max_area = area
                    largest_face = (x, y, w, h)

            if largest_face:
                x, y, w, h = largest_face
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Prepare face ROI
                h_frame, w_frame = frame.shape[:2]
                if x >= 0 and y >= 0 and x + w <= w_frame and y + h <= h_frame and w > 0 and h > 0:
                    face_roi = frame[y:y+h, x:x+w]
                    # Show the face ROI being sent to the model
                    face_resized = cv2.resize(face_roi, (224, 224))
                    cv2.imshow("Face ROI", face_resized)

                    # Only process a new emotion if cooldown passed and no thread running
                    if (emotion_thread is None or not emotion_thread.is_alive()) and (current_time - last_emotion_time > emotion_cooldown):
                        def emotion_worker(face_img):
                            try:
                                pil_face = self.preprocess_face(face_img)
                                inputs = self.processor(images=pil_face, return_tensors="pt")
                                with torch.no_grad():
                                    outputs = self.model(**inputs)
                                    predicted_class_id = int(torch.argmax(outputs.logits, dim=1))
                                    emotion = self.labels[predicted_class_id]
                                nonlocal emotion_result, last_emotion, last_emotion_time
                                emotion_result = emotion
                                last_emotion = emotion
                                last_emotion_time = time.time()
                                print(f"✅ Emotion detected: {emotion}")
                                # Improved streak and TTS logic
                                now = time.time()
                                if self.last_streak_emotion == emotion:
                                    if self.streak_start_time is None:
                                        self.streak_start_time = now
                                        self.streak_announced = False
                                    elif not self.streak_announced and (now - self.streak_start_time) >= 2.0:
                                        # Only speak if cooldown passed or emotion is new
                                        if (self.last_spoken_emotion != emotion) or (now - self.last_spoken_time > self.speak_cooldown):
                                            message = self.get_encouraging_message(emotion)
                                            print(f"[TTS] Speaking for emotion '{emotion}': {message}")
                                            self.speak_message(message)
                                            self.last_spoken_emotion = emotion
                                            self.last_spoken_time = now
                                        self.streak_announced = True
                                else:
                                    self.last_streak_emotion = emotion
                                    self.streak_start_time = now
                                    self.streak_announced = False
                            except Exception as e:
                                print(f"⚠️ Error processing emotion: {e}")
                                emotion_result = "Error"
                        emotion_thread = threading.Thread(target=emotion_worker, args=(face_resized,))
                        emotion_thread.start()

                    # Display the last detected emotion
                    display_emotion = last_emotion if last_emotion is not None else "Processing..."
                    cv2.putText(frame, f"Emotion: {display_emotion}", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # No face detected, close the ROI window if open
                try:
                    cv2.destroyWindow("Face ROI")
                except cv2.error:
                    pass

            # Calculate and display FPS
            fps_counter += 1
            if current_time - fps_start_time >= 1.0:
                self.fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time

            cv2.putText(frame, f"Faces: {len(faces)} | FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Facial Emotion Recognition", frame)

            # Add small delay to prevent excessive CPU usage
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()

    def display_encouraging_message(self, frame, message):
        """
        Display encouraging message on the frame with nice formatting
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Split message into lines if it's too long
        words = message.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= 40:  # Max 40 characters per line
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Display each line
        y_offset = height - 50 - (len(lines) * 30)
        for i, line in enumerate(lines):
            # Add background rectangle for better visibility
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_width, text_height = text_size
            
            # Background rectangle
            cv2.rectangle(frame, 
                         (10, y_offset + i * 30 - text_height - 5),
                         (10 + text_width + 20, y_offset + i * 30 + 5),
                         (0, 0, 0), -1)
            
            # Text
            cv2.putText(frame, line, (20, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Application closed successfully.")

def test_camera():
    try:
        
        import numpy as np
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as e:
        print(f"❌ Required package not found: {e.name}. Please install it using pip.")
        exit(1)
    cap = ThreadedVideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return ret

def main():
    print("=== Optimized Facial Expression Detection ===")
    if not test_camera():
        print("❌ Webcam not working.")
        return

    try:
        detector = OptimizedFacialExpressionDetector()
        detector.run_detection()
    except Exception as e:
        print(f"❌ Application error: {e}")
        try:
            detector.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()
