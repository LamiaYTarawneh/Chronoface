# Import Packages
import cv2
import os
import time
import numpy as np
import pickle
import sklearn
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import csv
import shutil
import matplotlib.pyplot as plt
from collections import defaultdict
import threading
from queue import Queue
import requests
#from flask import Flask, render_template, request, Response, jsonify
from flask import Flask, render_template, request, Response, redirect, url_for, session, flash, jsonify
from datetime import timedelta

import os

class FaceAttendanceSystem:
    def __init__(self, headless=False):
        self.headless = headless
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = "../attendance_system"

        self.dirs = {
            'employees': f"{self.base_dir}/employees",
            'embedding_labels': f"{self.base_dir}/embedding_labels",
            'visitors': f"{self.base_dir}/visitors",
            'logs': f"{self.base_dir}/logs",
            'session': f"{self.base_dir}/sessions/{self.session_id}",
            'models': f"{self.base_dir}/models",
            'frames': f"{self.base_dir}/sessions/{self.session_id}/frames",
            'visualizations': f"{self.base_dir}/sessions/{self.session_id}/visualizations"
        }

        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        self._init_face_detector()
        self._init_face_recognition()

        self.embeddings_file = f"{self.dirs['embedding_labels']}/embeddings.pkl"
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_metadata = []  # Stores additional info about each face
        self._load_embeddings()

        self.visitor_counter = self._get_last_visitor_id() + 1

        self.log_file = f"{self.dirs['logs']}/attendance_{self.session_id}.csv"
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Person ID', 'Name', 'Type', 'Action', 'Similarity'])

        # Increased recognition threshold for better confidence
        self.recognition_threshold = 0.65
        self.recognition_confidence = 0.85  # Higher threshold for positive recognition

        self.face_tracking = {}  # Tracks faces across frames
        self.face_detection_history = {}  # Track when faces were last detected
        self.recognized_faces = {}  # Stores recognized faces data

        # Increased minimum time between detections of the same face
        self.face_recapture_timeout = 120  # Seconds before recapturing the same face

        # Video capture settings
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        # Optimized video capture settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency

        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        self.frame_queue = Queue(maxsize=2)  # Queue for frame processing

        # Reduce terminal output frequency (every 30 seconds instead of 5)
        self.status_update_interval = 30

        # System settings
        self.auto_capture = True
        self.running = True
        self.processing_enabled = True

        # Start frame processing thread
        self.processing_thread = threading.Thread(target=self._frame_processor, daemon=True)
        self.processing_thread.start()

        print(f"Face Attendance System initialized. Session ID: {self.session_id}")
        if self.headless:
            print("Running in headless mode (no GUI).")

    def _init_face_detector(self):
        cascade_path = f"{self.dirs['models']}/haarcascade_frontalface_default.xml"

        if not os.path.exists(cascade_path):
            print("Downloading face detection model...")
            cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

            try:
                os.makedirs(os.path.dirname(cascade_path), exist_ok=True)
                response = requests.get(cascade_url)
                if response.status_code == 200:
                    with open(cascade_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded face detection model to {cascade_path}")
                else:
                    raise Exception(f"Failed to download model, status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("Please download the models manually and place them in the 'models' directory.")
                raise

        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise Exception(f"Error loading cascade classifier from {cascade_path}")

        print("Face detector initialized using Haar Cascade")

    def _init_face_recognition(self):
        model_file = f"{self.dirs['models']}/openface_nn4.small2.v1.t7"

        if not os.path.exists(model_file):
            print("Downloading face recognition model...")
            model_url = "https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7"

            try:
                os.makedirs(os.path.dirname(model_file), exist_ok=True)
                response = requests.get(model_url)
                if response.status_code == 200:
                    with open(model_file, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded face recognition model to {model_file}")
                else:
                    raise Exception(f"Failed to download model, status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("Please download the model manually and place it in the 'models' directory.")
                raise

        try:
            self.face_recognition_model = cv2.dnn.readNetFromTorch(model_file)
            print("Face recognition model initialized")
        except Exception as e:
            print(f"Error loading face recognition model: {e}")
            raise

    def _load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    if len(data) == 3:  # New format with metadata
                        self.known_face_encodings, self.known_face_names, self.known_face_metadata = data
                    else:  # Old format
                        self.known_face_encodings, self.known_face_names = data
                        self.known_face_metadata = [{} for _ in self.known_face_names]
                print(f"Loaded {len(self.known_face_names)} embeddings from {self.embeddings_file}")
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                self.known_face_encodings = []
                self.known_face_names = []
                self.known_face_metadata = []
        else:
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_metadata = []
            self._process_employee_images()  # Process employee images if no embeddings file exists

    def _save_embeddings(self):
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump((self.known_face_encodings, self.known_face_names, self.known_face_metadata), f)
        print(f"Saved {len(self.known_face_names)} embeddings to {self.embeddings_file}")

    def _get_last_visitor_id(self):
        visitor_dirs = [d for d in os.listdir(self.dirs['visitors']) if os.path.isdir(os.path.join(self.dirs['visitors'], d))]
        ids = [int(d.split('_')[1]) for d in visitor_dirs if d.startswith('VISITOR_') and d.split('_')[1].isdigit()]
        return max(ids) if ids else 0

    def _process_employee_images(self):
        print("Processing employee images...")
        employee_dir = self.dirs['employees']

        if not os.path.exists(employee_dir):
            print(f"Employee directory {employee_dir} does not exist. Creating it.")
            os.makedirs(employee_dir, exist_ok=True)
            print("Please add employee images to the directory before next run.")
            return

        for employee_name in os.listdir(employee_dir):
            employee_path = os.path.join(employee_dir, employee_name)
            if os.path.isdir(employee_path):
                print(f"Processing employee: {employee_name}")
                employee_images = [f for f in os.listdir(employee_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                if not employee_images:
                    print(f"No images found for employee: {employee_name}")
                    continue

                for img_file in employee_images:
                    img_path = os.path.join(employee_path, img_file)
                    image = cv2.imread(img_path)

                    if image is None:
                        print(f"Could not read image: {img_path}")
                        continue

                    faces = self._detect_faces(image)

                    if faces:
                        # Use the largest face found in the image
                        faces = sorted(faces, key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
                        x1, y1, x2, y2 = faces[0]
                        face_img = image[y1:y2, x1:x2]

                        try:
                            face_embedding = self._get_face_embedding(face_img)

                            # Check if this employee already has embeddings
                            existing_indices = [i for i, name in enumerate(self.known_face_names)
                                              if name == f"EMPLOYEE_{employee_name}"]

                            if existing_indices:
                                # Update existing embedding with the new one (average)
                                for idx in existing_indices:
                                    self.known_face_encodings[idx] = np.mean(
                                        [self.known_face_encodings[idx], face_embedding], axis=0
                                    )
                            else:
                                # Add new embedding
                                self.known_face_encodings.append(face_embedding)
                                self.known_face_names.append(f"EMPLOYEE_{employee_name}")
                                self.known_face_metadata.append({
                                    'type': 'employee',
                                    'source': img_path,
                                    'first_seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'last_seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })

                            print(f"Processed {img_file} for {employee_name}")
                        except Exception as e:
                            print(f"Error getting face embedding from {img_path}: {e}")
                    else:
                        print(f"No face found in {img_path}")

        self._save_embeddings()
        print(f"Processed {len(self.known_face_names)} embeddings for employees")

    def _detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Adjusted parameters for better face detection with less false positives
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,  # Increased from 5 to 6 for better confidence
            minSize=(60, 60),  # Set minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        face_locations = [(x, y, x + w, y + h) for (x, y, w, h) in faces]
        return face_locations

    def _get_face_embedding(self, face_img):
        face_img = cv2.resize(face_img, (96, 96))

        face_blob = cv2.dnn.blobFromImage(
            face_img, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False
        )

        self.face_recognition_model.setInput(face_blob)
        embedding = self.face_recognition_model.forward()

        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        return embedding

    def _compute_face_id(self, face_location):
        x1, y1, x2, y2 = face_location
        face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        face_size = (x2 - x1) * (y2 - y1)
        return f"{face_center[0]}_{face_center[1]}_{face_size}"

    def _find_nearest_tracked_face(self, face_location, max_distance=80):  # Reduced max distance
        curr_center_x = (face_location[0] + face_location[2]) // 2
        curr_center_y = (face_location[1] + face_location[3]) // 2

        min_distance = float('inf')
        nearest_id = None

        for face_id, face_data in self.face_tracking.items():
            tracked_center_x = (face_data['location'][0] + face_data['location'][2]) // 2
            tracked_center_y = (face_data['location'][1] + face_data['location'][3]) // 2

            distance = np.sqrt((curr_center_x - tracked_center_x)**2 + (curr_center_y - tracked_center_y)**2)

            if distance < min_distance and distance < max_distance:
                min_distance = distance
                nearest_id = face_id

        return nearest_id, min_distance

    def recognize_face(self, face_embedding):
        if len(self.known_face_encodings) == 0:
            return None, None, 0.0

        # Calculate similarities with all known faces
        similarities = [cosine_similarity([face_embedding], [known_encoding])[0][0]
                       for known_encoding in self.known_face_encodings]

        best_match_index = np.argmax(similarities)
        best_similarity = similarities[best_match_index]

        if best_similarity >= self.recognition_threshold:
            person_id = self.known_face_names[best_match_index]
            name = person_id
            if person_id.startswith("EMPLOYEE_"):
                name = person_id.replace("EMPLOYEE_", "")
            elif person_id.startswith("VISITOR_"):
                name = f"Visitor {person_id.split('_')[1]}"

            # Only return positive recognition if similarity is above confidence threshold
            if best_similarity >= self.recognition_confidence:
                return person_id, name, best_similarity
            else:
                # For similarity between threshold and confidence, mark as potential match
                return f"POTENTIAL_{person_id}", name, best_similarity

        return None, None, 0.0

    def process_new_face(self, frame, face_coords):
        x1, y1, x2, y2 = face_coords
        face_image = frame[y1:y2, x1:x2]

        # Generate a position-based identifier for the face
        face_position_id = f"{(x1+x2)//2}_{(y1+y2)//2}_{x2-x1}"

        # Check if we've recently processed a face at this position
        current_time = time.time()
        if face_position_id in self.face_detection_history:
            time_since_last_detection = current_time - self.face_detection_history[face_position_id]['time']
            if time_since_last_detection < self.face_recapture_timeout:
                # Return cached identity if we've seen this face recently
                cached_data = self.face_detection_history[face_position_id]
                return cached_data['person_id'], cached_data['name'], face_coords, cached_data['similarity']

        face_id, distance = self._find_nearest_tracked_face(face_coords)

        # If we have a nearby face that was recently seen, use its identity
        if face_id and distance < 60:  # Reduced distance threshold
            if (current_time - self.face_tracking[face_id]['last_seen']) < self.face_recapture_timeout:
                self.face_tracking[face_id]['location'] = face_coords
                self.face_tracking[face_id]['last_seen'] = current_time

                # Update detection history
                self.face_detection_history[face_position_id] = {
                    'time': current_time,
                    'person_id': self.face_tracking[face_id]['person_id'],
                    'name': self.face_tracking[face_id]['name'],
                    'similarity': self.face_tracking[face_id]['similarity']
                }

                return self.face_tracking[face_id]['person_id'], self.face_tracking[face_id]['name'], face_coords, self.face_tracking[face_id]['similarity']

        try:
            face_embedding = self._get_face_embedding(face_image)
            person_id, name, similarity = self.recognize_face(face_embedding)

            if person_id is None or person_id.startswith("POTENTIAL_"):
                # New visitor or uncertain match
                if person_id is None:
                    # Definitely new visitor
                    person_id = f"VISITOR_{self.visitor_counter}"
                    self.visitor_counter += 1
                    name = person_id
                    action = "NEW_VISITOR"
                else:
                    # Potential match - use the potential ID but mark as unconfirmed
                    person_id = person_id.replace("POTENTIAL_", "")
                    action = "POTENTIAL_MATCH"

                # Add to known faces if it's a new visitor
                if action == "NEW_VISITOR":
                    self.known_face_encodings.append(face_embedding)
                    self.known_face_names.append(person_id)
                    self.known_face_metadata.append({
                        'type': 'visitor',
                        'source': 'live_capture',
                        'first_seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'last_seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    self._save_embeddings()

                    visitor_dir = os.path.join(self.dirs['visitors'], person_id)
                    os.makedirs(visitor_dir, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    face_file = os.path.join(visitor_dir, f"{timestamp}.jpg")
                    cv2.imwrite(face_file, face_image)

                self._log_attendance(person_id, name, action, similarity)
                print(f"{action}: {person_id} (Similarity: {similarity:.4f})")
            else:
                # Confirmed recognition
                self._log_attendance(person_id, name, "RECOGNIZED", similarity)
                print(f"Recognized: {person_id} (Similarity: {similarity:.4f})")

                # Update visitor directory if this is a visitor
                if person_id.startswith("VISITOR_"):
                    visitor_dir = os.path.join(self.dirs['visitors'], person_id)
                    os.makedirs(visitor_dir, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    face_file = os.path.join(visitor_dir, f"{timestamp}.jpg")
                    cv2.imwrite(face_file, face_image)

            new_face_id = f"face_{len(self.face_tracking) + 1}"
            self.face_tracking[new_face_id] = {
                'person_id': person_id,
                'name': name,
                'location': face_coords,
                'last_seen': current_time,
                'embedding': face_embedding,
                'similarity': similarity
            }

            # Update detection history
            self.face_detection_history[face_position_id] = {
                'time': current_time,
                'person_id': person_id,
                'name': name,
                'similarity': similarity
            }

            return person_id, name, face_coords, similarity

        except Exception as e:
            print(f"Error processing face: {e}")
            return None, "Error", face_coords, 0.0

    def _log_attendance(self, person_id, name, action_type="DETECTED", similarity=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if person_id.startswith("EMPLOYEE_"):
            person_type = "EMPLOYEE"
            display_name = name.replace("EMPLOYEE_", "")
        else:
            person_type = "VISITOR"
            if person_id.startswith("VISITOR_"):
                display_name = f"Visitor {person_id.split('_')[1]}"
            else:
                display_name = name

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, person_id, display_name, person_type, action_type, similarity])

        print(f"Logged: {timestamp} | {person_id} | {display_name} | {person_type} | {action_type} | Similarity: {similarity}")

    def _cleanup_stale_tracking(self, max_age=10):
        current_time = time.time()
        stale_ids = []

        for face_id, face_data in self.face_tracking.items():
            if current_time - face_data['last_seen'] > max_age:
                stale_ids.append(face_id)

        for face_id in stale_ids:
            del self.face_tracking[face_id]

        return len(stale_ids)

    def _frame_processor(self):
        """Separate thread for processing frames to improve performance"""
        while self.running:
            if not self.frame_queue.empty() and self.processing_enabled:
                frame, face_locations = self.frame_queue.get()

                recognized_info = []
                current_time = time.time()

                # Process only distinct faces (avoid multiple detections of the same face)
                processed_positions = set()

                for face_location in face_locations:
                    # Create a position identifier for the face
                    x1, y1, x2, y2 = face_location
                    center_x, center_y = (x1+x2)//2, (y1+y2)//2
                    face_pos_id = f"{center_x//20}_{center_y//20}"  # Group similar positions

                    # Skip if we already processed a face in this position
                    if face_pos_id in processed_positions:
                        continue

                    processed_positions.add(face_pos_id)

                    face_id, distance = self._find_nearest_tracked_face(face_location)

                    if face_id and distance < 60:  # Reduced distance threshold
                        if (current_time - self.face_tracking[face_id]['last_seen']) < 1:
                            self.face_tracking[face_id]['location'] = face_location
                            self.face_tracking[face_id]['last_seen'] = current_time
                            recognized_info.append((
                                self.face_tracking[face_id]['person_id'],
                                self.face_tracking[face_id]['name'],
                                face_location,
                                self.face_tracking[face_id]['similarity']
                            ))
                            continue

                    if self.auto_capture:
                        result = self.process_new_face(frame, face_location)
                        if result:
                            person_id, name, bbox, similarity = result
                            recognized_info.append((person_id, name, bbox, similarity))

                # Update recognized faces for display
                self.recognized_faces = recognized_info

    def calculate_fps(self):
        """Calculate and return current FPS"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_update

        if elapsed >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time

        return self.fps

    def draw_info_on_frame(self, frame, face_locations):
        display_frame = frame.copy()
        fps = self.calculate_fps()

        for face_location in face_locations:
            x1, y1, x2, y2 = face_location
            label = "Unknown"
            color = (0, 165, 255)  # Orange for unknown

            # Check if this face was recognized in the current processing
            for person_id, name, bbox, similarity in self.recognized_faces:
                if bbox == face_location:
                    if person_id.startswith("EMPLOYEE_"):
                        color = (0, 255, 0)  # Green for employees
                        label = name.replace("EMPLOYEE_", "")
                    elif person_id.startswith("VISITOR_"):
                        color = (0, 255, 255)  # Yellow for visitors
                        label = f"Visitor {person_id.split('_')[1]}"
                    elif person_id.startswith("POTENTIAL_"):
                        color = (255, 255, 0)  # Light blue for potential matches
                        label = f"Maybe {name.replace('EMPLOYEE_', '')}"
                    break

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(display_frame, (x1, y2-35), (x2, y2), color, cv2.FILLED)
            cv2.putText(display_frame, label, (x1+6, y2-6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        # Status information
        status_texts = [
            f"FPS: {fps:.1f}",
            f"Faces: {len(face_locations)}",
            f"Auto-capture: {'ON' if self.auto_capture else 'OFF'}",
            f"Session: {self.session_id}",
            f"Known faces: {len(self.known_face_names)}"
        ]

        for i, text in enumerate(status_texts):
            cv2.putText(display_frame, text, (10, 30 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return display_frame

    def save_frame(self, frame, prefix="frame"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_path = os.path.join(self.dirs['frames'], f"{prefix}_{timestamp}.jpg")
        cv2.imwrite(frame_path, frame)
        return frame_path

    def run(self):
        print("\nFace Attendance System")
        print("Running indefinitely. Press Ctrl+C to stop.")

        start_time = time.time()
        save_frame_interval = 10
        last_frame_save = 0
        last_cleanup_time = time.time()
        cleanup_interval = 5
        last_status_update = 0  # Track last status update time

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error capturing frame")
                    break

                frame = cv2.flip(frame, 1)
                face_locations = self._detect_faces(frame)

                # Clean up stale tracking periodically
                current_time = time.time()
                if (current_time - last_cleanup_time) > cleanup_interval:
                    removed = self._cleanup_stale_tracking()
                    if removed > 0:
                        print(f"Removed {removed} stale face tracking entries")
                    last_cleanup_time = current_time

                # Add frame to processing queue if not full
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put((frame.copy(), face_locations))

                # Save frame periodically if faces detected
                if (current_time - last_frame_save) > save_frame_interval:
                    if len(face_locations) > 0:
                        annotated_frame = self.draw_info_on_frame(frame, face_locations)
                        saved_path = self.save_frame(annotated_frame)
                        print(f"Saved frame with {len(face_locations)} faces to {saved_path}")
                        last_frame_save = current_time

                display_frame = self.draw_info_on_frame(frame, face_locations)

                if not self.headless:
                    cv2.imshow("Face Attendance System", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('a'):
                        self.auto_capture = not self.auto_capture
                        print(f"Auto-capture {'enabled' if self.auto_capture else 'disabled'}")
                    elif key == ord('p'):
                        self.processing_enabled = not self.processing_enabled
                        print(f"Face processing {'enabled' if self.processing_enabled else 'disabled'}")

                # Print status every N seconds (increased interval)
                elapsed_seconds = int(time.time() - start_time)
                if elapsed_seconds > 0 and (current_time - last_status_update) >= self.status_update_interval:
                    elapsed_minutes = elapsed_seconds // 60
                    remaining_seconds = elapsed_seconds % 60
                    print(f"Status: FPS: {self.fps:.1f} | Faces: {len(face_locations)} | Running: {elapsed_minutes}m {remaining_seconds}s | Known faces: {len(self.known_face_names)}")
                    last_status_update = current_time

        except KeyboardInterrupt:
            print("\nUser interrupted the session")
        finally:
            self.running = False
            self.cap.release()
            if not self.headless:
                cv2.destroyAllWindows()

            # Wait for processing thread to finish
            if self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1)

            print(f"Session ended. Log saved to: {self.log_file}")
            self._generate_session_summary()
            self._generate_visualizations()

    def _generate_session_summary(self):
        summary_file = f"{self.dirs['session']}/summary.txt"

        attendance = {}
        with open(self.log_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 6:
                    person_id = row[1]
                    if person_id not in attendance:
                        attendance[person_id] = {
                            "name": row[2],
                            "type": row[3],
                            "count": 0,
                            "first_seen": row[0],
                            "last_seen": row[0],
                            "similarities": []
                        }
                    attendance[person_id]["count"] += 1
                    attendance[person_id]["last_seen"] = row[0]
                    attendance[person_id]["similarities"].append(float(row[5]))  # Store similarity

        with open(summary_file, 'w') as f:
            f.write(f"Session Summary: {self.session_id}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"Time: {datetime.now().strftime('%H:%M:%S')}\n\n")

            f.write("Attendance Overview:\n")
            f.write(f"Total unique individuals: {len(attendance)}\n")

            employees = {id: data for id, data in attendance.items() if data["type"] == "EMPLOYEE"}
            visitors = {id: data for id, data in attendance.items() if data["type"] == "VISITOR"}

            f.write(f"Employees present: {len(employees)}\n")
            f.write(f"Visitors: {len(visitors)}\n\n")

            f.write("Employee Details:\n")
            for person_id, data in employees.items():
                avg_similarity = sum(data['similarities']) / len(data['similarities'])
                f.write(f"- {data['name']}: Detected {data['count']} times, Avg Similarity: {avg_similarity:.4f}\n")
                f.write(f"  First seen: {data['first_seen']}, Last seen: {data['last_seen']}\n")

            f.write("\nVisitor Details:\n")
            for person_id, data in visitors.items():
                avg_similarity = sum(data['similarities']) / len(data['similarities'])
                f.write(f"- {person_id} (ID {person_id.split('_')[1]}): Detected {data['count']} times, Avg Similarity: {avg_similarity:.4f}\n")
                f.write(f"  First seen: {data['first_seen']}, Last seen: {data['last_seen']}\n")

        print(f"Session summary saved to: {summary_file}")
        shutil.copy(self.log_file, f"{self.dirs['session']}/attendance.csv")

    def _generate_visualizations(self):
        print("Generating session visualizations...")

        if not os.path.exists(self.log_file):
            print(f"Log file not found: {self.log_file}")
            return

        logs = []
        with open(self.log_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 6:
                    timestamp, person_id, name, person_type, action, similarity = row
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    logs.append({
                        'timestamp': timestamp,
                        'person_id': person_id,
                        'name': name,
                        'type': person_type,
                        'action': action,
                        'similarity': float(similarity)
                    })

        if not logs:
            print("No log entries found to visualize")
            return

        # 1. Attendance over time
        plt.figure(figsize=(12, 6))

        time_data = defaultdict(int)
        for entry in logs:
            time_key = entry['timestamp'].strftime("%H:%M")
            time_data[time_key] += 1

        times = sorted(time_data.keys())
        counts = [time_data[t] for t in times]

        plt.plot(times, counts, marker='o', linestyle='-')
        plt.title('Detection Events Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Detections')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.dirs['visualizations']}/detections_over_time.png")
        plt.close()

        # 2. Person type distribution
        plt.figure(figsize=(10, 6))

        person_types = [entry['type'] for entry in logs]
        type_counts = {t: person_types.count(t) for t in set(person_types)}

        plt.bar(type_counts.keys(), type_counts.values(), color=['#3498db', '#e74c3c'])
        plt.title('Distribution of Person Types')
        plt.xlabel('Person Type')
        plt.ylabel('Count')
        plt.savefig(f"{self.dirs['visualizations']}/person_type_distribution.png")
        plt.close()

        # 3. Top detected individuals
        plt.figure(figsize=(12, 8))

        person_data = defaultdict(int)
        for entry in logs:
            display_name = entry['name']
            if entry['type'] == 'EMPLOYEE':
                display_name = f"Employee: {display_name}"
            else:
                display_name = f"Visitor: {entry['person_id'].split('_')[1]}"
            person_data[display_name] += 1

        # Sort by count and take top 10
        top_people = sorted(person_data.items(), key=lambda x: x[1], reverse=True)[:10]
        people_names = [p[0] for p in top_people]
        people_counts = [p[1] for p in top_people]

        plt.barh(people_names, people_counts, color='#2ecc71')
        plt.title('Top Detected Individuals')
        plt.xlabel('Number of Detections')
        plt.ylabel('Person')
        plt.tight_layout()
        plt.savefig(f"{self.dirs['visualizations']}/top_detected_individuals.png")
        plt.close()

        # 4. Action types
        plt.figure(figsize=(10, 6))

        action_types = [entry['action'] for entry in logs]
        action_counts = {a: action_types.count(a) for a in set(action_types)}

        plt.pie(action_counts.values(), labels=action_counts.keys(), autopct='%1.1f%%',
                shadow=True, startangle=90, colors=['#9b59b6', '#f39c12', '#1abc9c'])
        plt.axis('equal')
        plt.title('Action Types Distribution')
        plt.savefig(f"{self.dirs['visualizations']}/action_types_distribution.png")
        plt.close()

        # Create an HTML report
        html_report = f"{self.dirs['visualizations']}/report.html"
        with open(html_report, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Attendance System - Session {self.session_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                    .chart {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9; }}
                    .chart img {{ max-width: 100%; height: auto; }}
                    .summary {{ margin-bottom: 20px; }}
                </style>
            </head>
            <body>
                <h1>Face Attendance System - Session Report</h1>
                <div class="summary">
                    <h2>Session Summary</h2>
                    <p><strong>Session ID:</strong> {self.session_id}</p>
                </div>
                <div class="dashboard">
                    <div class="chart">
                        <h2>Detection Events Over Time</h2>
                        <img src="detections_over_time.png" alt="Detection Events Over Time">
                    </div>
                    <div class="chart">
                        <h2>Person Type Distribution</h2>
                        <img src="person_type_distribution.png" alt="Person Type Distribution">
                    </div>
                    <div class="chart">
                        <h2>Top Detected Individuals</h2>
                        <img src="top_detected_individuals.png" alt="Top Detected Individuals">
                    </div>
                    <div class="chart">
                        <h2>Action Types Distribution</h2>
                        <img src="action_types_distribution.png" alt="Action Types Distribution">
                    </div>
                </div>
            </body>
            </html>
            """)

        print(f"Visualizations and report saved to: {self.dirs['visualizations']}")

#if __name__ == "__main__":
#    # Create the system and run indefinitely
#    system = FaceAttendanceSystem(headless=False)
#    system.run()


app = Flask(__name__)
app.secret_key = '1234'  # Change this in production!
app.permanent_session_lifetime = timedelta(minutes=30)

# Use the existing FaceAttendanceSystem instance
system = FaceAttendanceSystem(headless=True)

@app.route('/')
def index():
    return render_template('index.html')  # Will show live feed

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        success, frame = system.cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        face_locations = system._detect_faces(frame)

        recognized_info = []
        current_time = time.time()
        processed_positions = set()

        for face_location in face_locations:
            x1, y1, x2, y2 = face_location
            center_x, center_y = (x1+x2)//2, (y1+y2)//2
            face_pos_id = f"{center_x//20}_{center_y//20}"

            if face_pos_id in processed_positions:
                continue
            processed_positions.add(face_pos_id)

            if system.auto_capture:
                result = system.process_new_face(frame, face_location)
                if result:
                    person_id, name, bbox, similarity = result
                    recognized_info.append((person_id, name, bbox, similarity))

        system.recognized_faces = recognized_info
        annotated_frame = system.draw_info_on_frame(frame, face_locations)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == '1234':
            session['logged_in'] = True
            return redirect(url_for('attendance'))
        else:
            flash('Incorrect password. Try again.')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/attendance')
def attendance():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    attendance_data = []
    log_dir = system.dirs['logs']
    for filename in os.listdir(log_dir):
        if filename.startswith('attendance_') and filename.endswith('.csv'):
            with open(os.path.join(log_dir, filename), 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                attendance_data.extend(list(reader))
    return render_template('report.html', attendance_data=attendance_data)


# import base64
# import numpy as np
#
# @app.route('/identify', methods=['POST'])
# def identify_face():
#     try:
#         data = request.get_json()
#         image_data = data.get('image')
#
#         if not image_data:
#             return jsonify({'error': 'No image data provided'}), 400
#
#         # Decode base64 image
#         image_data = image_data.split(',')[1]  # Remove header
#         img_bytes = base64.b64decode(image_data)
#         nparr = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#         # Detect faces
#         faces = system._detect_faces(img)
#         if not faces:
#             return jsonify({'error': 'No face detected'}), 404
#
#         # For simplicity, use the largest face
#         faces = sorted(faces, key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
#         x1, y1, x2, y2 = faces[0]
#         face_img = img[y1:y2, x1:x2]
#
#         # Get embedding and check for match
#         embedding = system._get_face_embedding(face_img)
#         person_id, name, similarity = system.recognize_face(embedding)
#
#         if person_id is None:
#             return jsonify({
#                 'status': 'unknown',
#                 'message': 'No match found',
#                 'similarity': similarity
#             })
#         else:
#             return jsonify({
#                 'status': 'identified',
#                 'person_id': person_id,
#                 'name': name,
#                 'similarity': similarity
#             })
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

#@app.route('/identify_page')
#def identify_page():
#    return render_template('identify.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


