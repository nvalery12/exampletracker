import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F


def get_model():
    model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
    model.eval()
    return model


def detect_people(model, images):
    with torch.no_grad():
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)

        padded_images = []
        for img in images:
            pad_top = (max_h - img.shape[0]) // 2
            pad_bottom = max_h - img.shape[0] - pad_top
            pad_left = (max_w - img.shape[1]) // 2
            pad_right = max_w - img.shape[1] - pad_left

            padded_img = cv2.copyMakeBorder(
                img,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=0,
            )
            padded_images.append(padded_img)

        image_tensors = [F.to_tensor(img) for img in padded_images]
        image_tensors = torch.stack(image_tensors)

        predictions = model(image_tensors)

        detected_people_batch = []
        for i in range(len(predictions)):  # Iterate through batch
            boxes = predictions[i]["boxes"]
            scores = predictions[i]["scores"]
            labels = predictions[i]["labels"].cpu()  # Convert to CPU tensor
            persons = [
                box
                for box, score, label in zip(boxes, scores, labels)
                if score > 0.5 and label.item() == 1
            ]
            detected_people_batch.append(persons)

        return detected_people_batch


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Use the default webcam
    model = get_model()

    frame_buffer = []
    batch_size = 8

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)

        if len(frame_buffer) >= batch_size:
            batch_frames = frame_buffer[-batch_size:]
            detected_people_batch = detect_people(model, batch_frames)
            frame_buffer.clear()

            for detected_people, frame in zip(detected_people_batch, batch_frames):
                for box in detected_people:
                    x, y, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                cv2.imshow("Person Detection", frame)
                key = cv2.waitKey(1)
                if key == 27:  # Press 'Esc' to exit
                    break

    cap.release()
    cv2.destroyAllWindows()
