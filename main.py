import cv2
import dlib
import random
import threading
from deepface import DeepFace

# Load dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load dlib's correlation tracker
trackers = {}
colors = {}

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Set width
cap.set(4, 1080)  # height

# Open the video file for playback
video_file = "your_video.mp4"  # Replace with the video file path
video_cap = cv2.VideoCapture(video_file)

frame_count = 0


def video_display():
    while True:
        ret, video_frame = video_cap.read()
        if not ret:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, video_frame = video_cap.read()
        cv2.imshow("Video", video_frame)


# Start a thread for video display
video_thread = threading.Thread(target=video_display)
video_thread.start()

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

            # Analyze the emotion using deepface
            dominant_emotion = "None"
            try:
                emotion_analysis = DeepFace.analyze(
                    frame[y1 - 1 : y2 + 1, x1 - 1 : x2 + 1], actions=["emotion"]
                )

                # Get the dominant emotion from the emotion analysis result
                dominant_emotion = emotion_analysis[0]["dominant_emotion"]
            except:
                pass
            finally:
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

# Stop the video display thread
video_thread.join()

cap.release()
cv2.destroyAllWindows()
