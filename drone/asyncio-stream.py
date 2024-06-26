import asyncio

from threading import Thread

import cv2

from tello_asyncio import Tello, VIDEO_URL


print('[main thread] START')

def fly():
    async def main():
        drone = Tello()

        try:
            await asyncio.sleep(1)
            await drone.connect()
            # await drone.start_video(connect=False)
        except:
            pass