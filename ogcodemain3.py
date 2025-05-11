from ultralytics import YOLO
import cv2
import pyttsx3
import socket
import time

# --- YOLO and video setup ---
model = YOLO('yolo11n.pt')
video_path = 'http://192.168.113.121/stream'

FOCAL_LENGTH = 700  # example focal length in pixels
KNOWN_HEIGHT_PIXELS = 300  # assume approx pixel height for a person at 2 meters
KNOWN_DISTANCE = 2.0  # meters
NUM_REGIONS = 5
DISTANCE_THRESHOLD = 4  # meters

# Initialize text-to-speech
engine = pyttsx3.init()

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
region_width = frame_width // NUM_REGIONS

# --- ESP32 TCP socket setup ---
ESP32_IP = "192.168.60.121"
ESP32_PORT = 1234

try:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ESP32_IP, ESP32_PORT))
    print(f"✅ Connected to ESP32 at {ESP32_IP}:{ESP32_PORT}")
except Exception as e:
    print(f"❌ Could not connect to ESP32: {e}")
    client_socket = None

previous_decision = None
last_decision_time = 0

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        results = model(frame)
        region_obstacles = [[] for _ in range(NUM_REGIONS)]

        for r in results:
            for obj in r.boxes:
                x1, y1, x2, y2 = map(int, obj.xyxy[0])
                cls_name = model.names[int(obj.cls[0])]
                height = y2 - y1

                # Estimate distance based on a simple proportional relationship
                if height > 0:
                    distance = (KNOWN_HEIGHT_PIXELS * KNOWN_DISTANCE) / height
                    start_region = x1 // region_width
                    end_region = x2 // region_width
                    for region_index in range(start_region, min(end_region + 1, NUM_REGIONS)):
                        region_obstacles[region_index].append(distance)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{cls_name}: {distance:.2f}m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        for i in range(1, NUM_REGIONS):
            x_pos = i * region_width
            cv2.line(frame, (x_pos, 0), (x_pos, frame_height), (255, 255, 0), 2)

        # Decision-making logic
        min_distances = [min(region, default=float('inf')) for region in region_obstacles]
        if min_distances[2] < DISTANCE_THRESHOLD:
            if all(dist < DISTANCE_THRESHOLD for dist in min_distances):
                decision = "Obstacle ahead, stop!"
            elif min_distances[0] < DISTANCE_THRESHOLD or min_distances[1] < DISTANCE_THRESHOLD:
                decision = "Go right!"
            elif min_distances[3] < DISTANCE_THRESHOLD or min_distances[4] < DISTANCE_THRESHOLD:
                decision = "Go left!"
            else:
                decision = "Move left or right!"
        else:
            decision = "Path is clear!"

        if decision != previous_decision or (current_time - last_decision_time >= 5):
            print(f"Decision: {decision}")
            engine.say(decision)
            engine.runAndWait()
            last_decision_time = current_time
            previous_decision = decision

            if client_socket:
                try:
                    client_socket.send((decision + "\n").encode())
                except Exception as e:
                    print(f"Failed to send decision to ESP32: {e}")

        cv2.putText(frame, decision, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

if client_socket:
    client_socket.close()
    print("Socket closed.")

print("✅ Processing complete.")
