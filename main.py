import cv2
import dlib
import random
import numpy as np
from deepface import DeepFace
from deepface.basemodels import VGGFace
from deepface.commons import functions

# Load dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load dlib's correlation tracker
trackers = {}
colors = {}

# Load the emotion detection model
emotion_model = VGGFace.loadModel()
emotion_labels = functions.emotion_labels

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Set width
cap.set(4, 1080)  # height

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    if frame_count % 2 != 0:
        # Detect faces using dlib's face detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            new_trackers = {}

            for idx, face in enumerate(faces):
                x1, y1, x2, y2 = (
                    face.left(),
                    face.top(),
                    face.right(),
                    face.bottom(),
                )
                face_region = frame[y1:y2, x1:x2]

                if idx not in trackers:
                    # Initialize tracker for detected face
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x1, y1, x2, y2)
                    tracker.start_track(frame, rect)
                    new_trackers[idx] = tracker

                    # Generate a random color for each new person
                    colors[idx] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                else:
                    new_trackers[idx] = trackers[idx]

            trackers = new_trackers

        for idx, tracker in trackers.items():
            tracker.update(frame)
            pos = tracker.get_position()
            x1, y1, x2, y2 = (
                int(pos.left()),
                int(pos.top()),
                int(pos.right()),
                int(pos.bottom()),
            )
            color = colors[idx]

            # Analyze the emotion using the emotion detection model
            dominant_emotion = "None"
            try:
                face_for_emotion = cv2.resize(face_region, (48, 48))
                face_for_emotion = cv2.cvtColor(face_for_emotion, cv2.COLOR_BGR2GRAY)
                face_for_emotion = np.reshape(
                    face_for_emotion,
                    [1, face_for_emotion.shape[0], face_for_emotion.shape[1], 1],
                )
                emotion_analysis = emotion_model.predict(face_for_emotion)
                dominant_emotion = emotion_labels[np.argmax(emotion_analysis)]
            except:
                pass

            # Draw bounding box and display the dominant emotion analysis
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame,
                f"Person {idx} - Emotion: {dominant_emotion}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

        # Display the frame
        cv2.imshow("Face Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
