import math
import random
import time
import cv2
from line_profiler import profile

import numpy as np

from .pid import *

from .ram_helper import (
    check_wall,
    clamp,
    init_values,
    invert_y,
    mix_speed_turn,
    to_float
)


class Ram():
    # ----------------------------- CONSTANTS -----------------------------
    HISTORY_BUFFER = 20  # how many previous Huey or enemy position we are recording
    DANGER_ZONE = 55  # TODO: for smarter algo
    MAX_SPEED = 1  # magnitude between 0 and 1
    MAX_TURN = 1  # between 0 and 1
    ARENA_WIDTH = 700  # in pixels
    TOLERANCE = 10  # how close Huey's prev pos are permitted to be
    BACK_UP_SPEED = -0.7
    BACK_UP_TURN = 0
    FORWARD_SPEED = 0.7
    FORWARD_TURN = 0
    LEFT_SPEED = 0.7
    LEFT_TURN = -0.7
    RIGHT_SPEED = 0.7
    RIGHT_TURN = 0.7
    BACK_UP_THRESHOLD = 10  # > Double EDGE_THRESHOLD
    EDGE_THRESHOLD = 5
    RECOVERY_SPEED_VALUES = [
        BACK_UP_SPEED * 0.8, LEFT_SPEED * 0.8, RIGHT_SPEED * 0.8, FORWARD_SPEED * 0.8]
    RECOVERY_TURN_VALUES = [BACK_UP_TURN, LEFT_TURN, RIGHT_TURN, FORWARD_TURN]
    USE_PID = True
    is_recovering = False
    is_backing = False
    reverse = 0
    recovery_step = 0
    against_wall = ""
    moving_forward = -1

    def __init__(self, bots=None, huey_position=(np.array([ARENA_WIDTH, ARENA_WIDTH])), huey_old_position=(np.array([ARENA_WIDTH, ARENA_WIDTH])),
                 huey_orientation=45, enemy_position=np.array([0, 0]), huey_old_turn=0, huey_old_speed=0, is_recovering=False) -> None:
        # ----------------------------- INIT -----------------------------
        if bots is None:
            # initialize the position and orientation of huey
            self.huey_position = np.array(huey_position if huey_position is not None else (
                self.ARENA_WIDTH / 2, self.ARENA_WIDTH / 2), dtype=float)
            self.huey_old_position = np.array(
                huey_old_position if huey_old_position is not None else self.huey_position.copy(), dtype=float)
            # TODO: Fix orientation init
            self.huey_orientation = float(
                huey_orientation if huey_orientation is not None else 0.0)
            # initialize the current enemy position
            self.enemy_position = np.array(
                enemy_position if enemy_position is not None else (0.0, 0.0), dtype=float)
            self.huey_girth = 67

        else:
            self.huey_position = init_values(
                bots, self.ARENA_WIDTH, is_pos=True, is_huey=True)
            self.huey_old_position = init_values(
                bots, self.ARENA_WIDTH, is_pos=True, is_huey=True)
            self.huey_orientation = init_values(
                bots, self.ARENA_WIDTH, is_pos=False, is_huey=True)
            self.enemy_position = init_values(
                bots, self.ARENA_WIDTH, is_pos=True, is_huey=False)
            if bots["huey"] and len(bots["huey"]) > 0:
                self.huey_girth = (math.dist(bots['huey'].get('bbox')[
                                   1], bots['huey'].get('bbox')[0]))/2
            else:
                self.huey_girth = 67

        self.huey_old_speed = huey_old_speed
        self.huey_old_turn = huey_old_turn
        self.left = 0
        self.right = 0

        # initialize the enemy position array
        self.huey_pos_count = 1
        self.huey_previous_positions = []
        self.huey_previous_positions.append(self.huey_position)

        # initialize the enemy orientation array
        self.huey_orient_count = 1
        self.huey_previous_orientations = []
        self.huey_previous_orientations.append(self.huey_orientation)

        self.enemy_previous_positions = []
        self.enemy_previous_positions.append(self.enemy_position)

        # old time
        self.old_time = time.time()
        # delta time
        self.delta_t = 0.001

        # THIS WORKS IN TESTBOX
        self.turn_pid = PIDController(
            kp=0.008, ki=0.000, kd=0.0005, output_limits=(-1.0, 1.0))
        self.speed_pid = PIDController(
            kp=0.003, ki=0.000, kd=0.000, output_limits=(-1.0, 1.0))

        # Iter 1: Increase turn kp to account for not turning enough as we accelerate to robot
        # self.turn_pid = PIDController(kp=0.01, ki=0.000, kd=0.0005, output_limits=(-1.0, 1.0))
        # self.speed_pid = PIDController(kp=0.003, ki=0.000, kd=0.000, output_limits=(-1.0,1.0))

        # Iter 2: Turning derivative down to 0.003 for turn
        # self.turn_pid = PIDController(kp=0.008, ki=0.000, kd=0.0003, output_limits=(-1.0, 1.0))
        # self.speed_pid = PIDController(kp=0.003, ki=0.000, kd=0.000, output_limits=(-1.0,1.0))

        # Iter 3: Turning derivative down to 0.0015 for turn (no jitter, still over shooting a little)
        # self.turn_pid = PIDController(kp=0.008, ki=0.000, kd=0.00015, output_limits=(-1.0, 1.0))
        # self.speed_pid = PIDController(kp=0.003, ki=0.000, kd=0.000, output_limits=(-1.0,1.0))

        # Iter 4: Increasing proportional for turn
        # self.turn_pid = PIDController(kp=0.012, ki=0.000, kd=0.00015, output_limits=(-1.0, 1.0))
        # self.speed_pid = PIDController(kp=0.003, ki=0.000, kd=0.000, output_limits=(-1.0,1.0))

        # Iter 5: Decreasing derivative (see if it gets shaky)
        # self.turn_pid = PIDController(kp=0.012, ki=0.000, kd=0.00005, output_limits=(-1.0, 1.0))
        # self.speed_pid = PIDController(kp=0.003, ki=0.000, kd=0.000, output_limits=(-1.0,1.0))

        # Iter 6: Increasing derivative (got shakey)
        # self.turn_pid = PIDController(kp=0.012, ki=0.000, kd=0.0001, output_limits=(-1.0, 1.0))
        # self.speed_pid = PIDController(kp=0.003, ki=0.000, kd=0.000, output_limits=(-1.0,1.0))

        # Iter 7: Increasing deriative again
        # self.turn_pid = PIDController(kp=0.012, ki=0.000, kd=0.00015, output_limits=(-1.0, 1.0))
        # self.speed_pid = PIDController(kp=0.003, ki=0.000, kd=0.000, output_limits=(-1.0,1.0))

        # recovery
        self.recovering_until = 2.0
        self.recover_speed = 0.5
        self.recover_turn = 0.5
        self.is_recovering = False
        self.is_backing = False
    # ----------------------------- HELPER METHODS -----------------------------

    ''' use a PID controller to move the bot to the desired position '''

    def huey_move(self, speed: float, turn: float):
        speed = clamp(to_float(speed, 0.0), -1, 1)
        turn = clamp(to_float(turn,  0.0), -1, 1)

        left, right = mix_speed_turn(speed, turn)
        self.left, self.right = left, right
        return {'left': self.left, 'right': self.right, 'speed': speed, 'turn': turn}

    ''' moves Huey backwards, forward, left, right'''

    def recovery_sequence(self):
        self.recovery_step += 1
        duration = random.uniform(0.35, 0.6)
        self.recovering_until = time.time() + duration
        self.recover_speed = self.RECOVERY_SPEED_VALUES[self.recovery_step % 4]
        self.recover_turn = self.RECOVERY_TURN_VALUES[self.recovery_step % 4]

    # returns the list of previous orienations. to be used in corner detection
    def previous_orientations(self) -> list:
        return self.huey_previous_orientations

    def check_previous_position_and_orientation(self, can_recover: bool = True):
        if not can_recover:
            self.is_recovering = False
            self.is_backing = False
            self.moving_forward = 0
            return False

        counter_pos = 0
        x_curr, y_curr = self.huey_position

        for prev_pos in self.huey_previous_positions:
            if math.sqrt((x_curr - prev_pos[0])**2 + (y_curr - prev_pos[1])**2) < Ram.TOLERANCE:
                counter_pos += 1

        # for prev_orientation in self.huey_previous_orientations:
        #     # TODO: work out angle range
        #     if abs(prev_orientation - self.huey_orientation) < Ram.TOLERANCE * 0.5:
        #         counter_orientation += 1

        print("👨‍🔧👨‍🔧👨‍🔧 counter pos:", counter_pos)

        if counter_pos >= self.BACK_UP_THRESHOLD:
            self.is_recovering = True
            self.is_backing = False
            return True
        self.is_recovering = False
        self.is_backing = False
        return False

    def check_arena_edge(self, can_recover: bool = True):
        if not can_recover:
            self.is_recovering = False
            self.is_backing = False
            self.moving_forward = 0
            return False
        counter_pos = 0
        counter_orientation = 0
        x_curr, y_curr = self.huey_position

        for prev_pos in self.huey_previous_positions:
            if x_curr == prev_pos[0] and y_curr == prev_pos[1]:
                counter_pos += 1

        for prev_orientation in self.huey_previous_orientations:
            if prev_orientation == self.huey_orientation:
                counter_orientation += 1

        # print(f"💅POPOS:💅 {self.huey_position}")
        # print(f"🛸ORORIE:🛸 {self.huey_orientation}")
        # print(f"🦒🦒🦒GIRTH {self.huey_girth}")
        # print(f"🇦🇮COUNTER POS {counter_pos}")
        # print(f"😹COUNTER EDGE {counter_orientation}")

        self.reverse = 1

        if self.BACK_UP_THRESHOLD > counter_pos and counter_pos >= self.EDGE_THRESHOLD*2 and self.BACK_UP_THRESHOLD > counter_orientation and counter_orientation >= self.EDGE_THRESHOLD*2:
            self.reverse = -1

        if self.BACK_UP_THRESHOLD > counter_pos and counter_pos >= self.EDGE_THRESHOLD and self.BACK_UP_THRESHOLD > counter_orientation and counter_orientation >= self.EDGE_THRESHOLD:
            # Huey against left wall
            if (self.huey_position[0] < self.huey_girth):
                self.against_wall = "LEFT"
                if (0 <= self.huey_orientation < 45 or 315 < self.huey_orientation <= 359):
                    # print("👿 AGAINST A LEFT WALL, FORWARD 👿")
                    self.moving_forward = 1 * self.reverse
                    return 1 * self.reverse
                else:
                    # print("👼 AGAINST A LEFT WALL, BACK 👼")
                    self.moving_forward = -1 * self.reverse
                    return -1 * self.reverse

            # Huey against right wall
            elif self.huey_position[0] > 700 - self.huey_girth:
                self.against_wall = "RIGHT"
                if 135 < self.huey_orientation <= 225:
                    # print("🦋 AGAINST A RIGHT WALL, FORWARD 🦋")
                    self.moving_forward = 1 * self.reverse
                    return 1 * self.reverse
                else:
                    # print("🐛 AGAINST A RIGHT WALL, BACK 🐛")
                    self.moving_forward = -1 * self.reverse
                    return -1 * self.reverse

            # Huey against top wall
            elif self.huey_position[1] < self.huey_girth:
                self.against_wall = "TOP"
                if 225 < self.huey_orientation <= 315:
                    # print("🌝 AGAINST A TOP WALL, FORWARD 🌝")
                    self.moving_forward = 1 * self.reverse
                    return 1 * self.reverse
                else:
                    # print("🌚 AGAINST A TOP WALL, BACK 🌚")
                    self.moving_forward = -1 * self.reverse
                    return -1 * self.reverse

            # Huey against bottom wall
            elif self.huey_position[1] > 700 - self.huey_girth:
                self.against_wall = "BOTTOM"
                if 45 < self.huey_orientation <= 135:
                    # print("🦐 AGAINST A BOTTOM WALL, FORWARD 🦐")
                    self.moving_forward = 1 * self.reverse
                    return 1 * self.reverse
                else:
                    # print("🍤 AGAINST A BOTTOM WALL, BACK 🍤")
                    self.moving_forward = -1 * self.reverse
                    return -1 * self.reverse

            self.moving_forward = 0
            # print("NO BACKY FORY💀💀💀")
            return 0
        return 0

    ''' 
    Returns the predicted desired orientation angle of the bot given all parameters, NOTE: the positive direction is counterclockwise
    Precondition: our_position & enemy_position 
    '''

    def predict_desired_turn_and_speed(self):
        check_wall(self.enemy_position)
        enemy_future_position = self.enemy_position

        huey_position_copy = np.copy(self.huey_position)
        if np.linalg.norm(self.enemy_position - huey_position_copy) < Ram.DANGER_ZONE:
            enemy_future_position = self.enemy_position
            if np.array_equal(self.enemy_position, huey_position_copy):
                return (0, 0)

        if (np.array_equal(huey_position_copy, enemy_future_position)):
            return (0, 0)

        # return the angle in degrees
        huey_orientation_rad = np.radians(self.huey_orientation)
        orientation = np.array(
            [math.cos(huey_orientation_rad), math.sin(huey_orientation_rad)])
        enemy_future_position = invert_y(enemy_future_position)
        huey_position_invert = invert_y(huey_position_copy)
        direction = enemy_future_position - huey_position_invert

        # calculate the angle between the bot and the enemy
        ratio = np.dot(direction, orientation) / \
            (np.linalg.norm(direction) * np.linalg.norm(orientation))
        ratio = clamp(ratio, -1, 1)
        angle = np.degrees(np.arccos(ratio))
        sign = np.sign(np.cross(orientation, direction))

        # Raw errors
        error_angle = angle * sign
        distance = np.linalg.norm(direction)

        return error_angle, distance

    ''' main method for the ram ram algorithm that turns to face the enemy and charge towards it '''

    def ram_ram(self, bots: dict[str, any] = None, can_recover: bool = True, fps=120, key=None):
        if self.is_recovering or self.is_backing:
            self.HISTORY_BUFFER = fps/2
        else:
            self.HISTORY_BUFFER = fps*2
        self.BACK_UP_THRESHOLD = 0.75*self.HISTORY_BUFFER
        self.EDGE_THRESHOLD = 0.25*self.HISTORY_BUFFER

        if key == ord("r"):  # Press Q on keyboard to exit
            print("Recovery key r pressed.")
            self.huey_previous_positions = []
            self.huey_previous_orientations = []
            self.huey_previous_positions.append(self.huey_position)
            self.huey_previous_orientations.append(self.huey_orientation)

        # Changed from 5 to 1, TODO: recovery values need adjusted
        if self.huey_pos_count % 1 == 0:
            self.huey_previous_positions.append(self.huey_position)
            self.huey_previous_orientations.append(self.huey_orientation)

            # print(f'🥶🥶🥶 Huey Pos Count: {self.huey_pos_count}')
        self.huey_pos_count += 1
        self.huey_orient_count += 1

        # Save Huey's last 10 positions
        if len(self.huey_previous_positions) > self.HISTORY_BUFFER:
            self.huey_previous_positions = self.huey_previous_positions[int(
                len(self.huey_previous_positions)-self.HISTORY_BUFFER):]

        if len(self.huey_previous_orientations) > self.HISTORY_BUFFER:
            self.huey_previous_orientations = self.huey_previous_orientations[int(
                len(self.huey_previous_orientations)-self.HISTORY_BUFFER):]

        # If the array for enemy_previous_positions is full, then pop the first one
        self.enemy_previous_positions.append(self.enemy_position)

        if len(self.enemy_previous_positions) > self.HISTORY_BUFFER:
            self.enemy_previous_positions = self.enemy_previous_positions[int(
                len(self.enemy_previous_positions)-self.HISTORY_BUFFER):]

        if (bots and bots["huey"] and len(bots["huey"]) > 0):
            self.huey_position = np.array(bots['huey'].get('center'))
            self.huey_previous_positions.append(self.huey_position)

            if (bots["huey"].get("orientation") is not None):
                self.huey_orientation = bots['huey'].get('orientation')
                self.huey_previous_orientations.append(self.huey_orientation)
            else:
                self.huey_previous_orientations.append(
                    self.huey_previous_orientations[-1])
        else:
            self.huey_previous_positions.append(
                self.huey_previous_positions[-1])

        if time.time() < self.recovering_until:
            # print("Recovering...")
            return self.huey_move(self.recover_speed, self.recover_turn)
        else:
            self.recovering_until = 0

        # print(f"💅POPOS:💅 {self.huey_position}")
        # print(f"🛸ORORIE:🛸 {self.huey_orientation}")
        # print(f"🦒🦒🦒GIRTH {self.huey_girth}")

        backup = self.check_arena_edge(can_recover)
        if backup == 1:
            self.is_backing = True
            self.is_recovering = False
            return self.huey_move(self.FORWARD_SPEED, self.FORWARD_TURN)
        elif backup == -1:
            self.is_backing = True
            self.is_recovering = False
            return self.huey_move(self.BACK_UP_SPEED, self.BACK_UP_TURN)
        self.is_backing = False
        if (self.check_previous_position_and_orientation(can_recover)):
            # print("Start 🍀SPORADIH🍀🍀🍀")
            self.recovery_sequence()  # SEQUENCE
            return self.huey_move(self.recover_speed, self.recover_turn)
        else:
            self.recovery_step = 0

        if bots and bots["huey"] and len(bots["huey"]) > 0:
            self.huey_girth = (math.dist(bots['huey'].get('bbox')[
                               1], bots['huey'].get('bbox')[0]))/2

            self.delta_t = time.perf_counter() - self.old_time
            self.old_time = time.perf_counter()
        else:
            # print("Prev pos appended.")
            return self.huey_move(self.huey_old_speed, self.huey_old_turn)

        if bots["enemy"]:
            self.enemy_position = np.array(bots['enemy']['center'])
            error_angle, distance = self.predict_desired_turn_and_speed()

            if self.USE_PID and self.delta_t > 0:
                # 1. Calculate Turn using PID
                turn = self.turn_pid.update(error_angle, self.delta_t)

                # 2. Calculate Base Speed using PID
                ramming_distance = distance + 100  # 100 pixels is the "overshoot"
                base_speed = self.speed_pid.update(
                    ramming_distance, self.delta_t)

                # 3. Angle Attenuation (The "Weapon First" logic)
                clamped_angle = clamp(error_angle, -90, 90)
                angle_rad = math.radians(clamped_angle)

                # Using cosine gives a smooth curve. Squaring it makes the drop-off
                # sharper, heavily penalizing driving when not perfectly aligned.
                alignment_factor = math.cos(angle_rad) ** 2

                # Final speed is the PID speed scaled by how well we are aimed
                speed = base_speed * alignment_factor

            else:
                # Fallback if PID is off
                turn = clamp(error_angle * (Ram.MAX_TURN / 180.0), -1, 1)
                speed = 1 - (abs(error_angle) * (Ram.MAX_SPEED / 180.0))
                speed = clamp(speed, -1, 1)

            self.huey_old_turn, self.huey_old_speed = turn, speed
            return self.huey_move(speed, turn)

        else:
            # enemy bot not detected, previous position appended
            self.enemy_previous_positions.append(
                self.enemy_previous_positions[-1])
            self.enemy_position = self.enemy_previous_positions[-1]

            error_angle, distance = self.predict_desired_turn_and_speed()

            if self.USE_PID and self.delta_t > 0:
                turn = self.turn_pid.update(error_angle, self.delta_t)
                base_speed = self.speed_pid.update(
                    distance + 100, self.delta_t)
                alignment_factor = math.cos(
                    math.radians(clamp(error_angle, -90, 90))) ** 2
                speed = base_speed * alignment_factor
            else:
                turn = clamp(error_angle * (Ram.MAX_TURN / 180.0), -1, 1)
                speed = clamp(
                    1 - (abs(error_angle) * (Ram.MAX_SPEED / 180.0)), -1, 1)

            self.huey_old_turn, self.huey_old_speed = turn, speed
            return self.huey_move(speed, turn)
