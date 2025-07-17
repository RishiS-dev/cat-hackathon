import cv2
import mediapipe as mp
import time

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Eye landmarks (left and right)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def get_eye_aspect_ratio(landmarks, eye_idx, shape):
    # Calculate EAR for an eye
    image_w, image_h = shape
    points = [(int(landmarks[idx].x * image_w), int(landmarks[idx].y * image_h)) for idx in eye_idx]
    # EAR calculation: (|p2-p6| + |p3-p5|) / (2*|p1-p4|)
    import numpy as np
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# EAR threshold and fatigue time
EAR_THRESHOLD = 0.25
CLOSED_EYE_TIME = 2.0  # seconds

cap = cv2.VideoCapture(0)
eye_closed_start = None
fatigue_alert = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    h, w = frame.shape[:2]
    fatigue_alert = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = get_eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_IDX, (w, h))
            right_ear = get_eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_IDX, (w, h))
            avg_ear = (left_ear + right_ear) / 2.0

            # Draw eyes for visualization
            for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0,255,0), -1)

            if avg_ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif time.time() - eye_closed_start > CLOSED_EYE_TIME:
                    fatigue_alert = True
            else:
                eye_closed_start = None

    # Show warning
    if fatigue_alert:
        cv2.putText(frame, "WARNING: Fatigue Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    else:
        cv2.putText(frame, "Driver Monitoring Active", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Driver Fatigue Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()