import cv2
from djitellopy import Tello
from flask import Flask, Response
from ultralytics import YOLO

from drone.config import TELLO_HOST

tello = Tello(host=TELLO_HOST)

model = YOLO('yolov8n.pt')
app = Flask(__name__)


@app.route('/')
def home():
    return Response("This flask home")


def video_feed():
    def generate():
        while True:
            result_frame = tello.get_frame_read()
            frame = result_frame.frame

            results = model.track(frame, persist=True, verbose=False)

            predict_image = results[0].plot()

            ret, jpeg = cv2.imencode(".jpg", predict_image)

            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    return Response(generate(), mimetype="multipart/x-mixed-replace;boundary=frame")


if __name__ == '__main__':

    tello.connect()

    print(tello.get_battery())
    tello.streamon()

    try:

        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        print(f'An error occured in Flask App: {e}')

    finally:
        tello.streamoff()
        tello.end()

    # while True:
    #     result_frame = tello.get_frame_read()
    #     frame = result_frame.frame
    #
    #     # frame = cv2.resize(frame, (320, 320))
    #
    #     # adjusted_image = adjust_color_balance(frame, red_gain=1, green_gain=1.1, blue_gain=0.5)
    #     adjusted_image = frame
    #
    #     results = model.track(adjusted_image, persist=True, verbose=True)
    #
    #     predict_image = results[0].plot()
    #
    #     print(result_frame.frame)
    #     cv2.imshow("Drone Camera", adjusted_image)
    #     cv2.imshow("Prediction", predict_image)
    #
    #     # print(result)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break
    #
    # cv2.destroyAllWindows()
