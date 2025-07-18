import cv2
import mediapipe as mp
import time
import numpy as np

class FatigueDetector:
    def __init__(self):
        # --- Constants ---
        self.EAR_THRESHOLD = 0.25
        self.CLOSED_EYE_TIME = 2.0  # seconds
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

        # --- MediaPipe Setup ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # --- State Variables ---
        self.eye_closed_start = None
        self.fatigue_alert = False

    def _get_eye_aspect_ratio(self, landmarks, eye_idx, shape):
        """Calculates Eye Aspect Ratio for a single eye."""
        image_w, image_h = shape
        points = [(int(landmarks[idx].x * image_w), int(landmarks[idx].y * image_h)) for idx in eye_idx]
        
        A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
        C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
        
        return (A + B) / (2.0 * C)

    def process_frame(self, frame):
        """Processes a single video frame to detect fatigue."""
        h, w = frame.shape[:2]
        
        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        self.fatigue_alert = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear = self._get_eye_aspect_ratio(face_landmarks.landmark, self.LEFT_EYE_IDX, (w, h))
                right_ear = self._get_eye_aspect_ratio(face_landmarks.landmark, self.RIGHT_EYE_IDX, (w, h))
                avg_ear = (left_ear + right_ear) / 2.0

                # --- Fatigue Logic ---
                if avg_ear < self.EAR_THRESHOLD:
                    if self.eye_closed_start is None:
                        self.eye_closed_start = time.time()
                    elif time.time() - self.eye_closed_start > self.CLOSED_EYE_TIME:
                        self.fatigue_alert = True
                else:
                    self.eye_closed_start = None
        
        # --- Draw Alert on Frame ---
        if self.fatigue_alert:
            cv2.putText(frame, "WARNING: Fatigue Detected!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "Driver Monitoring Active", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
        return frame