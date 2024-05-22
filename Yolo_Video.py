from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    cap = cv2.VideoCapture(path_x)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("../YOLO-Weights/yolov8n.pt")
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name} {conf}'

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                # Calculate text size for the label
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 10
                # Draw filled rectangle for label background
                cv2.rectangle(img, (x1, y1), c2, (255, 0, 255), -1)
                # Put label text
                cv2.putText(img, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        yield img

    cap.release()
    cv2.destroyAllWindows()
