import serial
import json

ser = serial.Serial("COM3", 115200, timeout=1)  # use /dev/ttyUSB0 on Linux
ser.write(b"get_imu\n")
line = ser.readline().decode('utf-8').strip()

while (True):
    print(ser.readline().decode('utf-8').strip())
    json_string = ser.readline().decode('utf-8').strip()
    dict = json.loads(json_string)
    print(dict)