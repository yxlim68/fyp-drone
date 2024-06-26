import time

from djitellopy import Tello

tello = Tello()

tello.connect()

tello.connect_to_wifi("YourMama", "S0upn!fy")

while True:
    print(tello.get_battery())

    time.sleep(1)