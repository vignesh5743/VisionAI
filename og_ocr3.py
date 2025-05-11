import cv2
import numpy as np
import time
import os
import shutil
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
import pyttsx3

# --- CONFIGURABLE PARAMETERS ---
ESP32_IP = '192.168.113.121'  # <-- Replace with your ESP32's IP address
STREAM_URL = f'http://{ESP32_IP}/stream'
COMMAND_PORT = 1234  # (not used in this basic capture version but reserved)
USE_ESP32_STREAM = True  # True = use ESP32, False = use local video

LOCAL_VIDEO_PATH = 'paragraph-moving.mov'  # fallback video
CAPTURE_DURATION = 7  # seconds to capture frames

pytesseract.pytesseract.tesseract_cmd = r'C:/Users/harih/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'

# --- Scoring Weights ---
SHARPNESS_WEIGHT = 0.4
CONTRAST_WEIGHT = 0.3
ILLUMINATION_WEIGHT = 0.1
COVERAGE_WEIGHT = 0.1
SKEW_WEIGHT = 0.1
DEBUG = False

# --- Helper Functions ---
def calculate_laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.var(laplacian)

def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def calculate_illumination_evenness(image, block_size=32):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    block_variances = [np.var(gray[y:y+block_size, x:x+block_size]) 
                       for y in range(0, gray.shape[0] - block_size + 1, block_size) 
                       for x in range(0, gray.shape[1] - block_size + 1, block_size)]
    return np.std(block_variances) if block_variances else 0

def calculate_page_coverage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.contourArea(largest_contour) / (image.shape[0] * image.shape[1])
    return 0

def calculate_skew_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angles = [np.degrees(theta) - 90 if 45 < np.degrees(theta) < 135 else np.degrees(theta) 
              for line in lines for rho, theta in [line[0]]] if lines is not None else []
    return np.median(angles) if angles else 0

def normalize(value, min_value, max_value):
    return 0.0 if max_value == min_value else (value - min_value) / (max_value - min_value)

def score_image(image):
    sharpness = calculate_laplacian_variance(image)
    contrast = calculate_contrast(image)
    illumination = calculate_illumination_evenness(image)
    coverage = calculate_page_coverage(image)
    skew = abs(calculate_skew_angle(image))

    SHARPNESS_MIN, SHARPNESS_MAX = 10, 1000
    CONTRAST_MIN, CONTRAST_MAX = 5, 60
    ILLUMINATION_MIN, ILLUMINATION_MAX = 2, 25
    COVERAGE_MIN, COVERAGE_MAX = 0.1, 0.9
    SKEW_MAX = 15

    sharpness_norm = normalize(sharpness, SHARPNESS_MIN, SHARPNESS_MAX)
    contrast_norm = normalize(contrast, CONTRAST_MIN, CONTRAST_MAX)
    illumination_norm = 1 - normalize(illumination, ILLUMINATION_MIN, ILLUMINATION_MAX)
    coverage_norm = normalize(coverage, COVERAGE_MIN, COVERAGE_MAX)
    skew_norm = 1 - min(normalize(skew, 0, SKEW_MAX), 1.0)

    if DEBUG:
        print(f"Sharpness: {sharpness:.2f}, Contrast: {contrast:.2f}, Illumination: {illumination:.2f}, Coverage: {coverage:.2f}, Skew: {skew:.2f}")
        print(f"Score: {sharpness_norm:.2f}, {contrast_norm:.2f}, {illumination_norm:.2f}, {coverage_norm:.2f}, {skew_norm:.2f}")

    total_score = (SHARPNESS_WEIGHT * sharpness_norm +
                   CONTRAST_WEIGHT * contrast_norm +
                   ILLUMINATION_WEIGHT * illumination_norm +
                   COVERAGE_WEIGHT * coverage_norm +
                   SKEW_WEIGHT * skew_norm)
    return total_score

def display_text_on_image(image, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, color=(0, 255, 0), thickness=2):
    cv2.putText(image, text, position, font, font_scale, color, thickness)

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say(text)
    engine.runAndWait()

def summarize_text(text, max_sentences=3):
    sentences = text.split('. ')
    if len(sentences) <= max_sentences:
        return text
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    top_sentence_indices = sentence_scores.argsort()[-max_sentences:][::-1]
    return ' '.join([sentences[i] for i in sorted(top_sentence_indices)])

# --- Main Program ---
def main():
    capture_dir = 'captures'
    if os.path.exists(capture_dir):
        shutil.rmtree(capture_dir)
    os.makedirs(capture_dir)

    # Select input source
    if USE_ESP32_STREAM:
        print(f"Connecting to ESP32 stream: {STREAM_URL}")
        cap = cv2.VideoCapture(STREAM_URL)
    else:
        print(f"Opening local video file: {LOCAL_VIDEO_PATH}")
        cap = cv2.VideoCapture(LOCAL_VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    best_score = -1
    best_frame = None
    best_frame_number = 0
    frame_count = 0
    scores_dict = {}

    start_time = time.time()

    print(f"Capturing frames for {CAPTURE_DURATION} seconds...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        elapsed_time = time.time() - start_time
        if elapsed_time > CAPTURE_DURATION:
            print("Capture time completed.")
            break

        frame_count += 1
        filename = os.path.join(capture_dir, f'frame_{frame_count}.png')
        cv2.imwrite(filename, frame)

        score = score_image(frame)
        scores_dict[f'frame_{frame_count}'] = score

        if score > best_score or best_frame is None:
            best_score = score
            best_frame = frame.copy()
            best_frame_number = frame_count

        display_text_on_image(frame, f"Score: {score:.2f}", (10, 30))
        display_text_on_image(frame, f"Best Score: {best_score:.2f}", (10, 60))
        display_text_on_image(frame, f"Time left: {max(0, int(CAPTURE_DURATION - elapsed_time))}s", (10, 90), color=(0, 255, 255))

        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit key pressed.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if best_frame is not None:
        extracted_text = pytesseract.image_to_string(best_frame)
        summarized_text = summarize_text(extracted_text)

        print(f"\nBest Score: {best_score:.2f} (Frame: {best_frame_number})")
        print("\n--- Extracted Text ---")
        print(extracted_text)
        print("\n--- Summarized Text ---")
        print(summarized_text)

        text_to_speech(summarized_text)

        print("\n--- All Frame Scores ---")
        for frame, score in scores_dict.items():
            print(f"{frame}: {score:.2f}")
    else:
        print("No frames captured.")

if __name__ == "__main__":
    main()
