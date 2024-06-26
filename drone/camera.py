import threading
import time
import cv2
from djitellopy import Tello
from ultralytics import YOLO
from plyer import notification
import numpy as np
import os

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

def snap(frame, confidence):
    if not os.path.exists('snapshots'):
        os.makedirs('snapshots')
    filename = f"snapshots/human_detected_{time.strftime('%Y%m%d_%H%M%S')}_{confidence:.2f}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Snapshot saved as {filename}")

def fly(tello):
    tello.move_up(50)
    while True:
        tello.move_forward(100)
        tello.rotate_clockwise(90)
        time.sleep(1)

if __name__ == '__main__':
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
                send_notification(f"Human detected with confidence {confidence:.2f}")
                snap(frame, confidence)  # Take a snapshot

        cv2.imshow("Drone Camera", frame)
        cv2.imshow("Prediction", predict_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    tello.land()
