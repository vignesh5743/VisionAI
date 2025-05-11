import cv2
import numpy as np
import os
import time
import socket
import pyttsx3
import matplotlib.pyplot as plt

# --- CONFIG ---
ESP32_IP = '192.168.221.121'  # Replace with your ESP32 IP
STREAM_URL = f'http://{ESP32_IP}:80/stream'
COMMAND_PORT = 1234

USE_ESP32_STREAM = True  # True = use ESP32 live stream, False = use local image
LOCAL_TEST_IMAGE = "b2/test/images1.jpeg"  # fallback test image if ESP32 not available

CAPTURE_DURATION = 7  # seconds to capture

# --- Functions ---
def load_landmark_descriptors(descriptor_folder):
    descriptors = {}
    orb = cv2.ORB_create(nfeatures=2000)

    for subfolder in os.listdir(descriptor_folder):
        subfolder_path = os.path.join(descriptor_folder, subfolder)

        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_path = os.path.join(subfolder_path, file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    keypoints, des = orb.detectAndCompute(image, None)

                    if des is not None:
                        descriptors[f"{subfolder}/{file}"] = (image, keypoints, des)

    return descriptors

def recognize_landmark(input_image, descriptors, ratio_test_threshold=0.75):
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    keypoints_input, input_des = orb.detectAndCompute(gray, None)

    if input_des is None or len(input_des) == 0:
        print("Warning: No keypoints detected")
        return "No keypoints detected"

    best_match = None
    best_match_count = 0

    for landmark, (image, keypoints, des) in descriptors.items():
        matches = bf.knnMatch(input_des, des, k=2)
        good_matches = [m for m, n in matches if m.distance < ratio_test_threshold * n.distance]

        if len(good_matches) > best_match_count:
            best_match_count = len(good_matches)
            best_match = landmark

    if best_match:
        best_match = best_match.split('/')[0]  # only folder name (currency)

    return best_match if best_match else "No currency recognized"

def speak_result(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def send_result_to_esp32(result_text, esp32_ip, esp32_port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((esp32_ip, esp32_port))
            s.sendall(result_text.encode())
            print(f"Sent to ESP32: {result_text}")
    except Exception as e:
        print(f"Failed to send result to ESP32: {e}")

# --- Main ---
def main():
    landmark_folder = "b2/landmarks"
    landmark_descriptors = load_landmark_descriptors(landmark_folder)

    if USE_ESP32_STREAM:
        print(f"Connecting to ESP32 stream: {STREAM_URL}")
        cap = cv2.VideoCapture(STREAM_URL)
    else:
        print(f"Loading local image: {LOCAL_TEST_IMAGE}")
        test_image = cv2.imread(LOCAL_TEST_IMAGE)
        if test_image is None:
            print("Failed to load test image.")
            return

    if USE_ESP32_STREAM:
        if not cap.isOpened():
            print("Error: Could not open ESP32 video stream")
            return

        start_time = time.time()
        best_frame = None
        print(f"Capturing for {CAPTURE_DURATION} seconds...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            elapsed = time.time() - start_time
            if elapsed > CAPTURE_DURATION:
                print("Capture complete.")
                break

            best_frame = frame  # Keep updating (latest frame)

            cv2.imshow("ESP32 Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit key pressed.")
                break

        cap.release()
        cv2.destroyAllWindows()

    # Choose final frame to recognize
    final_frame = best_frame if USE_ESP32_STREAM else test_image

    if final_frame is not None:
        recognized_label = recognize_landmark(final_frame, landmark_descriptors)

        print(f"\nRecognized Currency: {recognized_label}")

        speak_result(f"The recognized currency is {recognized_label}")

        send_result_to_esp32(recognized_label, ESP32_IP, COMMAND_PORT)

    else:
        print("No frame captured.")

if __name__ == "__main__":
    main()
