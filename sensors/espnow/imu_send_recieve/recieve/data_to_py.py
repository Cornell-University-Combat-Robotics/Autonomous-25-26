import serial
import json
import time


windows = "COM3"
mac = "/dev/tty.usbserial-0001"
ser = serial.Serial(mac, 115200, timeout=1)  # use /dev/ttyUSB0 on Linux
ser.write(b"get_imu\n")
line = ser.readline().decode('utf-8').strip()

while (True):
    #print(ser.readline().decode('utf-8').strip())
    try:
        json_string = ser.readline().decode('utf-8').strip()
        dict = json.loads(json_string)
        print(f"dict: {dict}")
    except:
        pass
    #time.sleep(0.5)