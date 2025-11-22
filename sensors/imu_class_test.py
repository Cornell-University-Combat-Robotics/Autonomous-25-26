import serial
import json
import time
import math
import imu_class as imu_class

windows = "COM3"
mac = "/dev/tty.usbserial-0001"

sensor = imu_class.IMU_sensor()

while True:
    try:
        print(f"yaw: {sensor.get_yaw_continuous()}")
        # print(f"is upside down: {sensor.is_upside_down()}")
    except KeyboardInterrupt:
        break 
    except Exception as e:
        print(e)
        continue
