import cv2
import numpy as np
import requests
from datetime import datetime

# Prompt user to enter the server IP address for sending data
server_ip = input("Enter the server IP address: ")
SERVER_URL = f"http://{server_ip}:80/update_count"  # Server endpoint for sending count data

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection
NMS_THRESHOLD = 0.4         # Non-maxima suppression threshold
MIN_BOX_AREA = 100          # Minimum area of detection box to consider
PERSON_CLASS_ID = 0         # COCO class ID for 'person'

# Load YOLO model and configurations
net = cv2.dnn.readNet("yolo-coco/yolov3-tiny.weights", "yolo-coco/yolov3-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class labels for detection
with open("yolo-coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def send_count_to_server(person_count, timestamp, latency_ms):
    """
    Sends the person count, timestamp, and latency data to the server.
    """
    try:
        payload = {
            "count": person_count,
            "timestamp": timestamp,
            "latency_ms": latency_ms
        }
        response = requests.post(SERVER_URL, json=payload)
        if response.status_code != 200:
            print(f"Failed to send count. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while sending data: {e}")

def process_frame(frame):
    """
    Processes a single frame to detect persons, apply NMS, and count persons.
    """
    start_time = datetime.utcnow().timestamp()
    height, width = frame.shape[:2]
    
    # Prepare the image for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get layer names and forward pass for detections
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)
    
    # Initialize lists for bounding boxes, confidences, and class IDs
    boxes, confidences, class_ids = [], [], []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD and class_id == PERSON_CLASS_ID:
                # Scale bounding box back to the original image size
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                
                # Calculate top-left corner for the bounding box
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply Non-Maxima Suppression to filter overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    person_count = len(indices)  # Count persons based on final boxes
    
    # Draw bounding boxes and labels on the frame
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Color for bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Calculate latency and send data to server
    end_time = datetime.utcnow().timestamp()
    latency_ms = (end_time - start_time) * 1000
    timestamp = datetime.utcnow().timestamp()
    send_count_to_server(person_count, timestamp, latency_ms)
    
    # Display person count and latency on the frame
    cv2.putText(frame, f"People Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Latency: {latency_ms:.2f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open the camera.")
        return
    print("[INFO] Starting camera...")
    
    # Process frames in real-time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture image from camera.")
            break
        frame = process_frame(frame)
        cv2.imshow("Camera Feed", frame)
        
        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera processing stopped.")

if __name__ == "__main__":
    main()
