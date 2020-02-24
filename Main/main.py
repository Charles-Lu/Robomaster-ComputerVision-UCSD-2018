import sys
sys.path.append("..")

from Camera.video import video_capture
from threading import Thread
from Aiming.detect import detect_main
import time


def main():
    memory = {}
    camera = Thread(target=video_capture, args=(memory, False, False, False))
    detect = Thread(target=detect_main, args=(memory, "cnn"))
    detect.start()
    time.sleep(20)
    camera.start()
    detect.join()


if __name__ == '__main__':
    main()
