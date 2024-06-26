from djitellopy import Tello

from drone.config import TELLO_HOST

tello = Tello(host=TELLO_HOST)

tello.connect()

tello.streamon()


try:
    while True:
        frame = tello.get_frame_read()
        print(frame)
except Exception as e:
    raise e
finally:
    tello.streamoff()
    tello.end()