import serial
import json
import time
import math

windows = "COM3"
mac = "/dev/tty.usbserial-0001"
ser = serial.Serial(mac, 115200, timeout=1)  # use /dev/ttyUSB0 on Linux
ser.write(b"get_imu\n")
line = ser.readline().decode('utf-8').strip()


def quaternion_to_euler(q_w, q_x, q_y, q_z):
    # Roll (x-axis rotation)
    roll = math.atan2(2 * (q_w * q_x + q_y * q_z), 1 - 2 * (q_x**2 + q_y**2))
    
    # Pitch (y-axis rotation)
    pitch = math.asin(2 * (q_w * q_y - q_z * q_x))
    
    # Yaw (z-axis rotation)
    yaw = math.atan2(2 * (q_w * q_z + q_x * q_y), 1 - 2 * (q_y**2 + q_z**2))
    
    return roll, pitch, yaw


while (True):
    #print(ser.readline().decode('utf-8').strip())
    try:
        json_string = ser.readline().decode('utf-8').strip()
        dict = json.loads(json_string)
        roll, pitch, yaw = quaternion_to_euler(dict["rotation"]["r"], dict["rotation"]["i"], dict["rotation"]["j"], dict["rotation"]["k"])
        groll, gpitch, gyaw = quaternion_to_euler(dict["game"]["r"], dict["game"]["i"], dict["game"]["j"], dict["game"]["k"])
        #print(f"dict: {dict}")
        yaw_to_deg = (yaw / math.pi) * 180
        gyaw_to_deg = (gyaw / math.pi) * 180
        print(f"yaw: {yaw_to_deg}")
        print(f"game yaw: {gyaw_to_deg}")
        # print(f"gravity: {dict["accelerometer"]["gravity_z"]}")
    except KeyboardInterrupt:
        break 
    except:
        continue
    #time.sleep(0.5)