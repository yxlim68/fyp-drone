from djitellopy import Tello

tello = Tello(host="192.168.8.100")

tello.connect()

tello.takeoff()

try:
    pass
except:
    pass
finally:
    tello.land()