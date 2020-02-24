import serial
import time


class Connection:
    def __init__(self, port):
        self.car = serial.Serial(port, 9600)
        print("Serial Port Connected")

    def send(self, pose):
        x, y = pose
        self.car.write((str(-x)[:4]+"\r\n").encode("ascii"))
        self.car.write((str(y)[:4] + "!\r\n").encode("ascii"))
        # print(str(-x)[:4], str(y)[:4])
        with open("wave.txt", "a") as f:
            f.write(str(pose[0])+","+str(pose[1])+"\n")
