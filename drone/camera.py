import threading
import time
import cv2
from djitellopy import Tello
from ultralytics import YOLO
from plyer import notification
import numpy as np
import os
from geopy.geocoders import Nominatim
import requests
import mysql.connector as connector
import platform
import subprocess
import sys

# Assuming a frame size of 960x720 (adjust according to your Tello camera resolution)
FRAME_WIDTH = 960
FRAME_HEIGHT = 720

# Check if XAMPP is running and start it if not
def check_and_start_xampp():
    if platform.system() == 'Windows':
        # Check if Apache is running
        apache_status = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq httpd.exe'], stdout=subprocess.PIPE)
        mysql_status = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq mysqld.exe'], stdout=subprocess.PIPE)

        if b'No tasks' in apache_status.stdout or b'No tasks' in mysql_status.stdout:
            print("Starting XAMPP...")
            subprocess.run(['C:\\xampp\\xampp_start.exe'])
            time.sleep(10)  # Wait for XAMPP to start
        else:
            print("XAMPP is already running.")
    else:
        print("Unsupported OS")
        sys.exit(1)

# Stop XAMPP after the script finishes
def stop_xampp():
    if platform.system() == 'Windows':
        print("Stopping XAMPP...")
        subprocess.run(['C:\\xampp\\xampp_stop.exe'])
    elif platform.system() == 'Linux':
        print("Stopping XAMPP...")
        subprocess.run(['/opt/lampp/lampp', 'stop'])

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def adjust_color_balance(image, red_gain=1.0, green_gain=1.0, blue_gain=1.0):
    b, g, r = cv2.split(image)
    r = cv2.multiply(r, red_gain)
    g = cv2.multiply(g, green_gain)
    b = cv2.multiply(b, blue_gain)
    return cv2.merge((b, g, r))

def gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def send_notification(message):
    notification.notify(
        title="Detection",
        message=message,
        timeout=5  # Duration in seconds
    )


# Snap function to save image and save to database
def snap(frame, confidence, x, y, w, h):
    # Save image to a directory served by XAMPP (htdocs)
    snapshot_dir = 'C:\\xampp\\htdocs\\snapshots'  # Adjust the path if needed
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    filename = f"human_detected_{time.strftime('%Y%m%d_%H%M%S')}_{confidence:.2f}.jpg"
    filepath = os.path.join(snapshot_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"Snapshot saved as {filepath}")

    # Encode image to bytes
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()

    # Draw bounding box on the image
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    save_to_database(confidence, image_bytes)

    # Generate download link
    download_url = f"http://localhost/snapshots/{filename}"
    print(f"Download URL: {download_url}")
    send_notification(f"Snapshot available at {download_url}")

def save_to_database(confidence, image_bytes):
    # Database connection details
    db_config = {
        'user': 'root',
        'password': '',
        'host': 'localhost',
        'database': 'drone',
        'raise_on_warnings': True
    }

    try:
        cnx = connector.connect(**db_config)
        cursor = cnx.cursor()

        add_detection = ("INSERT INTO img "
                         "(Confidence, SS) "
                         "VALUES (%s, %s)")
        data_detection = (confidence, image_bytes)

        cursor.execute(add_detection, data_detection)
        cnx.commit()

        cursor.close()
        cnx.close()
        print("Detection data saved to database.")
    except connector.Error as err:
        print(f"Error: {err}")

def fly(tello):
    tello.move_up(50)  # Move up by 50 cm
    while True:
        tello.move_forward(100)
        tello.rotate_clockwise(90)
        time.sleep(1)

def get_location(max_retries=3):
    url = 'https://api64.ipify.org?format=json'
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()
            return data['loc'].split(',')
        except (requests.ConnectionError, requests.HTTPError) as e:
            print(f"Error fetching location: {e}")
            retries += 1
            time.sleep(1)  # Wait for 1 second before retrying
    return None

if __name__ == '__main__':
    check_and_start_xampp()
    tello = Tello()
    model = YOLO('yolov8n.pt')
    tello.connect()
    tello.takeoff()
    print("Battery:", tello.get_battery())
    tello.streamon()

    fly_thread = threading.Thread(target=fly, args=(tello,), daemon=True)
    fly_thread.start()

    while True:
        result_frame = tello.get_frame_read()
        frame = result_frame.frame

        # Apply white balance
        frame = white_balance(frame)
        # Adjust color balance (optional, fine-tune as needed)
        frame = adjust_color_balance(frame, red_gain=1.1, green_gain=1.1, blue_gain=1.0)
        # Apply gamma correction
        frame = gamma_correction(frame, gamma=1.2)
        # Enhance contrast
        frame = enhance_contrast(frame)

        results = model(frame)

        predict_image = results[0].plot()

        # Check if any human is detected
        for result in results[0].boxes:
            if result.cls == 0:  # class 0 is typically 'person' in COCO dataset
                confidence = result.conf.item()  # Convert Tensor to Python float
                x1, y1, x2, y2 = map(int, result.xyxy[0])  # Get bounding box coordinates
                x, y, w, h = x1, y1, x2 - x1, y2 - y1  # Convert to (x, y, w, h)
                #
                # # Calculate approximate center of the bounding box
                # center_x = x + w // 2
                # center_y = y + h // 2

                # Get current location
                # location = get_location()
                # if location:
                #     latitude, longitude = location

                send_notification(f"Human detected with confidence {confidence:.2f}")
                snap(frame, confidence, x, y, w, h)  # Take a snapshot

                    # print(f"Human detected at Latitude: {latitude}, Longitude: {longitude}")

        cv2.imshow("Drone Camera", frame)
        cv2.imshow("Prediction", predict_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    tello.land()
    stop_xampp()
