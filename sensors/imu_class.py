import serial
import json
import time
import math

class IMUReadError(Exception):
    """Base exception for IMU serial read issues"""
    pass


class IMU_sensor():

    def __init__(self, port = None, baud_rate = 115200, timeout = 1):
        """
        Parameters
        ----------
        port : string, optional
            port esp32 is connected to. If None, then calls choose_port for port selection
        buad_rate : int, optional
            the baud_rate (data transmission rate)
        timeout : int, optional
            amount of time to wait before raising an error on the serial connection
        """   
        
        if port is None:
            port = self.choose_port()
        self.ser = serial.Serial(port, baud_rate, timeout=timeout)
        self.dict = {}
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        time.sleep(2)  # Wait for the serial connection to initialize
    
    def choose_port(self):
        """ 
        Allows user to determine what port the esp32 is on

        User Guide: 
        1. Look at port list printed by choose_port
        2. unplug esp32 and press '0' to refresh list
        3. Look to see which port is missing
        4. replug esp32 and refresh port list
        6. select index of esp32 port

        Returns: string port value (ex. "COM3")
        """

        def get_ports():
            available_ports = serial.tools.list_ports.comports()
            port_dic = {}
            if len(available_ports) == 0:
                print("No ports found")
            else:
                print("Choose a port from the options below:")
                for i in range(len(available_ports)):
                    port = available_ports[i]
                    port_dic[str(i+1)] = port.device
                    print(str(i+1) + ":", port)
            print("Choose 0 to refresh your options")

            selection = input("Enter your selection here: ")
            return [selection, port_dic]

        def check_validity(selection):
            while selection != "0" and selection not in port_dic:
                print("Selection invalid. Choose one of the following or 0 to refresh options:",
                      list(port_dic.keys()))
                selection = input("Enter your selection here: ")
            return selection

        selection, port_dic = get_ports()
        selection = check_validity(selection)

        while (selection == '0'):
            selection, port_dic = get_ports()
            selection = check_validity(selection)

        return port_dic[selection]

    def get_dict(self):
        """
        Updates dict field with the latest read (TODO: add try/except to raise IMUReadError)
        """
        json_string = self.ser.readline().decode('utf-8').strip()
        self.dict = json.loads(json_string)
    
    def is_upside_down(self):
        """
        Returns: -1 if bot is upside down and 1 if the bot is right side up
        """
        self.get_dict()
        return 1 if self.dict["accelerometer"]["gravity_z"] >= 0 else -1

    def get_yaw(self):
        """
        Returns imu orientation from 0-360 (TODO: Make it actually do that)
        Raises
        """
        # try to get a new reading for yaw
        self.get_dict()
        _, _, yaw = self.quaternion_to_euler(self.dict["game"]["r"], self.dict["game"]["i"], self.dict["game"]["j"], self.dict["game"]["k"])
        print(yaw)
        self.yaw = (yaw / math.pi) * 180
        return f'{time.time()} : {self.yaw}'
            
    

    def quaternion_to_euler(self, q_w, q_x, q_y, q_z):
        # Roll (x-axis rotation)
        self.roll = math.atan2(2 * (q_w * q_x + q_y * q_z), 1 - 2 * (q_x**2 + q_y**2))
    
        # Pitch (y-axis rotation)
        self.pitch = math.asin(2 * (q_w * q_y - q_z * q_x))
    
        # Yaw (z-axis rotation)
        self.yaw = math.atan2(2 * (q_w * q_z + q_x * q_y), 1 - 2 * (q_y**2 + q_z**2))
        
        if self.yaw < 0:
            self.yaw += 360        
            
        return self.roll, self.pitch, self.yaw