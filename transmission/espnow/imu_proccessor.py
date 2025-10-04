import serial
ser = serial.Serial("COM3", 115200, timeout=1)  # use /dev/ttyUSB0 on Linux
ser.write(b"get_imu\n")
line = ser.readline().decode('utf-8').strip()
print("IMU Data:", line)