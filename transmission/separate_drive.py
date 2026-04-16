from motors import Motor
from serial_conn import OurSerial
import time


LEFT_CHANNEL = 3
RIGHT_CHANNEL = 1

LEFT_INVERT = False
RIGHT_INVERT = True


class SkidDriver:
    """Skid-steer driver with one motor per channel and per-motor invert flags."""

    def __init__(
        self,
        ser: OurSerial,
        drive_speed: float = 0.5,
        turn_speed: float = 0.5,
        left_channel: int = LEFT_CHANNEL,
        right_channel: int = RIGHT_CHANNEL,
        left_invert: bool = LEFT_INVERT,
        right_invert: bool = RIGHT_INVERT,
    ):
        self.left_motor = Motor(ser, speed=0, channel=left_channel)
        self.right_motor = Motor(ser, speed=0, channel=right_channel)
        self.left_sign = -1 if left_invert else 1
        self.right_sign = -1 if right_invert else 1
        self.drive_speed = drive_speed
        self.turn_speed = turn_speed

    def _drive(self, left: float, right: float):
        self.left_motor.move(self.left_sign * left)
        self.right_motor.move(self.right_sign * right)

    def forward(self, duration: float = 0, speed: float = None):
        s = abs(speed if speed is not None else self.drive_speed)
        self._drive(s, s)
        if duration > 0:
            time.sleep(duration)

    def backward(self, duration: float = 0, speed: float = None):
        s = abs(speed if speed is not None else self.drive_speed)
        self._drive(-s, -s)
        if duration > 0:
            time.sleep(duration)

    def left(self, duration: float = 0, speed: float = None):
        s = abs(speed if speed is not None else self.turn_speed)
        self._drive(-s, s)
        if duration > 0:
            time.sleep(duration)

    def right(self, duration: float = 0, speed: float = None):
        s = abs(speed if speed is not None else self.turn_speed)
        self._drive(s, -s)
        if duration > 0:
            time.sleep(duration)

    def stop(self, t: float = 0):
        self._drive(0, 0)
        if t > 0:
            time.sleep(t)


if __name__ == "__main__":
    ser = OurSerial()
    drive = SkidDriver(ser, drive_speed=0.4, turn_speed=0.4)

    drive.forward(duration=1.5)
    drive.stop(t=0.5)

    drive.backward(duration=1.5)
    drive.stop(t=0.5)

    drive.left(duration=1.0)
    drive.stop(t=0.5)

    drive.right(duration=1.0)
    drive.stop(t=0.5)

    ser.cleanup()
