import cv2
import dlib
import random
import numpy as np
from deepface import DeepFace

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel"
)

# Load dlib's correlation tracker
trackers = {}
colors = {}

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Set width
cap.set(4, 1080)  # height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects using MobileNet SSD
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward()

    if len(detections) > 0:
        new_trackers = {}

        for idx in range(detections.shape[2]):
            confidence = detections[0, 0, idx, 2]

            if confidence > 0.5:
                class_id = int(detections[0, 0, idx, 1])

                if class_id == 15:
                    box = detections[0, 0, idx, 3:7] * np.array(
                        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                    )
                    (startX, startY, endX, endY) = box.astype("int")
                    pedestrian = dlib.rectangle(startX, startY, endX, endY)

                    if idx not in trackers:
                        # Initialize tracker for detected pedestrian
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(frame, pedestrian)
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

        # Extract the face region if within frame boundaries
        if y1 >= 0 and y2 < frame.shape[0] and x1 >= 0 and x2 < frame.shape[1]:
            face_region = frame[y1:y2, x1:x2]

            # Analyze the emotion using deepface
            dominant_emotion = "None"
            try:
                emotion_analysis = DeepFace.analyze(face_region, actions=["emotion"])

                # Get the dominant emotion from the emotion analysis result
                dominant_emotion = emotion_analysis[0]["dominant_emotion"]
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
    cv2.imshow("Pedestrian Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
