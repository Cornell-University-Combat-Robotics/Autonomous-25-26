from motors import Motor
from serial_conn import OurSerial
import time


THROTTLE_CHANNEL = 1
STEERING_CHANNEL = 3


class Driver:

    def __init__(self, ser: OurSerial, drive_speed: float = 0.5, turn_speed: float = 0.5):
        self.throttle = Motor(ser, speed=0, channel=THROTTLE_CHANNEL)
        self.steering = Motor(ser, speed=0, channel=STEERING_CHANNEL)
        self.drive_speed = drive_speed
        self.turn_speed = turn_speed

    def forward(self, duration: float = 0, speed: float = None):
        self.throttle.move(speed if speed is not None else self.drive_speed)
        if duration > 0:
            time.sleep(duration)

    def backward(self, duration: float = 0, speed: float = None):
        s = speed if speed is not None else self.drive_speed
        self.throttle.move(-abs(s))
        if duration > 0:
            time.sleep(duration)

    def left(self, duration: float = 0, speed: float = None):
        s = speed if speed is not None else self.turn_speed
        self.steering.move(-abs(s))
        if duration > 0:
            time.sleep(duration)

    def right(self, duration: float = 0, speed: float = None):
        self.steering.move(speed if speed is not None else self.turn_speed)
        if duration > 0:
            time.sleep(duration)

    def stop(self, t: float = 0):
        self.throttle.stop()
        self.steering.stop(t)


if __name__ == "__main__":
    ser = OurSerial()
    drive = Driver(ser, drive_speed=0.5, turn_speed=0.5)

    drive.forward(duration=1.5)
    drive.stop(t=0.5)

    drive.backward(duration=1.5)
    drive.stop(t=0.5)

    drive.left(duration=1.0)
    drive.stop(t=0.5)

    drive.right(duration=1.0)
    drive.stop(t=0.5)

    ser.cleanup()
