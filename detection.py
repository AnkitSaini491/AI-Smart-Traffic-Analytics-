from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

VEHICLES = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}


def detect_vehicles(source):
    cap = cv2.VideoCapture(source)

    vehicle_count = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, verbose=False)

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if class_name in VEHICLES:
                    vehicle_count[class_name] += 1

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0]
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        class_name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        cv2.imshow("Traffic Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return vehicle_count
