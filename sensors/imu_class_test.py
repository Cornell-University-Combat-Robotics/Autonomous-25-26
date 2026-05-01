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
        # sensor.get_yaw_continuous()
        print(f"yaw: {sensor.get_yaw_continuous()}")
        # print(f"is upside down: {sensor.get_upside_down_continuous()}")
        # print(f"Z: {sensor.get_field_continuous("gyroscope", "z")}")
    except KeyboardInterrupt:
        break 
    except Exception as e:
        print(e)
        continue

