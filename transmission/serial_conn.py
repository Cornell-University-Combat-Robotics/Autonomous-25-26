import time
import serial
import serial.tools.list_ports


class OurSerial():
    """
    The OurSerial module is used to establish connection with the arduino over serial communication

    Attributes
    ----------
    ser : serial.Serial
        a serial.Serial object that sets up the connection to the arduino

    Methods
        -------
        choose_port()
            used for user to select port for serial connection

        send_data(channel, speed)
            Sets [channel] to [speed]

        cleanup()
            Closes serial connection

    """

    def __init__(self, baudrate=115200, timeout=1, port=None, auto_select_iouusb=False):
        """
        Parameters
        ----------
        buadrate : int, optional
            the baudrate (data transmission rate)
        timeout : int, optional
            amount of time to wait before raising an error on the serial connection
        port : string, optional
            port arduino is connected to. If None, then calls choose_port for port selection
        """

        if port is None and auto_select_iouusb:
            port = self.auto_select_iouusb_port()
        if port is None:
            port = self.choose_port()
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # Wait for the serial connection to initialize

    @staticmethod
    def _port_search_text(port):
        fields = [
            port.device,
            port.name,
            port.description,
            port.hwid,
            port.manufacturer,
            port.product,
            port.interface,
            port.location,
            str(port),
        ]
        return " ".join(str(field) for field in fields if field)

    @classmethod
    def _transmission_port_score(cls, port):
        text = cls._port_search_text(port).lower()
        score = 0

        if "iousb" in text:
            score += 100
        if "arduino" in text:
            score += 40
        if "usbmodem" in text or "usbserial" in text:
            score += 25
        if "cu.usb" in text or "tty.usb" in text:
            score += 10
        if "bluetooth" in text:
            score -= 100

        return score

    @classmethod
    def auto_select_iouusb_port(cls):
        """
        Selects the Arduino-style USB serial port used for transmission.

        On macOS, PySerial often shows Arduino USB devices with an IOUSB*
        description/HWID string while the actual serial path is /dev/cu.usb*.
        """
        available_ports = list(serial.tools.list_ports.comports())
        scored_ports = [
            (cls._transmission_port_score(port), port)
            for port in available_ports
        ]
        candidates = [
            (score, port)
            for score, port in scored_ports
            if score > 0 and "iousb" in cls._port_search_text(port).lower()
        ]

        if not candidates:
            print("AUTOSELECT_TRANSMISSION did not find an IOUSB transmission port.")
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, selected_port = candidates[0]
        print(f"Auto-selected transmission port: {selected_port.device} ({selected_port})")
        return selected_port.device

    def choose_port(self):
        """ 
        Allows user to determine what port the arduino is on

        User Guide: 
        1. Look at port list printed by choose_port
        2. unplug arduino and press '0' to refresh list
        3. Look to see which port is missing
        4. replug arduino and refresh port list
        6. select index of arduino port

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

    def send_data(self, channel1, speed1, channel2=None, speed2=None):
        """
        Sets one or two channels to specified speeds.

        Parameters
        ----------
        channel1 : int
            First channel to set
        speed1 : float
            Value for first channel (-1.0 to 1.0)
        channel2 : int, optional
            Second channel to set
        speed2 : float, optional
            Value for second channel (-1.0 to 1.0)
        """
        data = f"{channel1} {speed1:.2f}"
        if channel2 is not None and speed2 is not None:
            data += f" {channel2} {speed2:.2f}"
        data += "\n"
        self.ser.write(data.encode())

    def cleanup(self):
        """Closes serial connection"""
        self.ser.close()
