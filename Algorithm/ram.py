import math
import time
import numpy as np
import random
from .ram_helper import invert_y, check_wall, clamp, to_float, mix_speed_turn


class Ram():
    # ----------------------------- CONSTANTS -----------------------------
    HISTORY_BUFFER = 10  # how many previous Huey or enemy position we are recording
    DANGER_ZONE = 55
    MAX_SPEED = 1  # magnitude between 0 and 1
    MAX_TURN = 1  # between 0 and 1
    ARENA_WIDTH = 1200  # in pixels
    TOLERANCE = 10  # how close Huey's prev pos are permitted to be
    BACK_UP_SPEED = -1
    BACK_UP_TURN = 0
    FORWARD_SPEED = 1
    FORWARD_TURN = 0
    LEFT_SPEED = 1
    LEFT_TURN = -1
    RIGHT_SPEED = 1
    RIGHT_TURN = 1
    BACK_UP_THRESHOLD = 5  # TODO: lower number of stagnant frames to trigger Huey back up?
    RECOVERY_SPEED_VALUES = [BACK_UP_SPEED, FORWARD_SPEED, LEFT_SPEED, RIGHT_SPEED]
    RECOVERY_TURN_VALUES = [BACK_UP_TURN, FORWARD_TURN, LEFT_TURN, RIGHT_TURN]
    USE_PID = True
    recovery_step = 0

    def __init__(self, bots=None, huey_position=(np.array([ARENA_WIDTH, ARENA_WIDTH])), huey_old_position=(np.array([ARENA_WIDTH, ARENA_WIDTH])),
                 huey_orientation=45, enemy_position=np.array([0, 0]), huey_old_turn=0, huey_old_speed=0) -> None:
        # ----------------------------- INIT -----------------------------
        if bots is None:
            print("hi 1")
            # initialize the position and orientation of huey
            self.huey_position = np.array(huey_position if huey_position is not None else (self.ARENA_WIDTH / 2, self.ARENA_WIDTH / 2), dtype=float)
            self.huey_old_position = np.array(huey_old_position if huey_old_position is not None else self.huey_position.copy(), dtype=float)
            # TODO: Fix orientation init
            self.huey_orientation = float(huey_orientation if huey_orientation is not None else 0.0)
            # initialize the current enemy position
            self.enemy_position = np.array(enemy_position if enemy_position is not None else (0.0, 0.0), dtype=float)
        else:
            print("hi 2")
            self.huey_position = np.array(bots['huey']['center'] if np.array(bots['huey']['center']) is not None else (self.ARENA_WIDTH / 2, self.ARENA_WIDTH / 2), dtype=float)
            print(f"CURR POS: {self.huey_position}")
            self.huey_old_position = np.array(bots['huey']['center'] if np.array(bots['huey']['center']) is not None else (self.ARENA_WIDTH / 2, self.ARENA_WIDTH / 2), dtype=float)
            print(f"OLD POS: {self.huey_old_position}")
            self.huey_orientation = float(bots['huey']['orientation'] if bots['huey']['orientation'] is not None else 0.0)
            print(f"ORIENTATION: {self.huey_orientation}")
            self.enemy_position = np.array(bots['enemy']['center'] if bots['enemy']['center'] is not None else (0.0, 0.0), dtype=float)
            print(f"ENEMY POS: {self.enemy_position}")

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

        #recovery
        self.recovering_until = 2.0
        self.recover_speed = 0.5
        self.recover_turn = 0.5
    # ----------------------------- HELPER METHODS -----------------------------

    ''' use a PID controller to move the bot to the desired position '''
    def huey_move(self, speed: float, turn: float):
        speed = clamp(to_float(speed, 0.0), -1, 1)
        turn  = clamp(to_float(turn,  0.0), -1, 1)

        left, right = mix_speed_turn(speed, turn)
        self.left, self.right = left, right
        return {'left': self.left, 'right': self.right, 'speed': speed, 'turn': turn}

    ''' moves Huey backwards, forward, left, right'''
    def recovery_sequence(self):
        print("—————————— RECOVERY_SEQUENCE. self.recovery_step: " + str(self.recovery_step))
        self.recovery_step += 1
        duration = random.uniform(0.5, 1.0)
        self.recovering_until = time.time() + duration
        self.recover_speed = self.RECOVERY_SPEED_VALUES[self.recovery_step%4]
        self.recover_turn = self.RECOVERY_TURN_VALUES[self.recovery_step%4]

    def check_previous_position_and_orientation(self):
        print("—————————— CHECK_PREVIOUS_POSITION_AND_ORIENTATION")
        counter_pos, counter_orientation = 0, 0
        x_curr, y_curr = self.huey_position

        for prev_pos in self.huey_previous_positions:
            if math.sqrt((x_curr - prev_pos[0])**2 + (y_curr - prev_pos[1])**2) < Ram.TOLERANCE:
                counter_pos += 1

        for prev_orientation in self.huey_previous_orientations:
            # TODO: work out angle range
            if abs(prev_orientation - self.huey_orientation) < Ram.TOLERANCE * 0.5:
                counter_orientation += 1

        if counter_pos >= Ram.BACK_UP_THRESHOLD and counter_orientation >= Ram.BACK_UP_THRESHOLD:
            print("check_previous_position_and_orientation returns true")
            return True
        
        print("check_previous_position_and_orientation returns false")
        return False

    ''' 
    Returns the predicted desired orientation angle of the bot given all parameters, NOTE: the positive direction is counterclockwise
    Precondition: our_position & enemy_position 
    '''
    def predict_desired_turn_and_speed(self):
        print("—————————— PREDICT_DESIRED_TURN_AND_SPEED")
        
        check_wall(self.enemy_position, 729)
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
        orientation = np.array([math.cos(huey_orientation_rad), math.sin(huey_orientation_rad)])
        enemy_future_position = invert_y(enemy_future_position)
        huey_position_invert = invert_y(huey_position_copy)
        direction = enemy_future_position - huey_position_invert
        
        # calculate the angle between the bot and the enemy
        ratio = np.dot(direction, orientation) / \
            (np.linalg.norm(direction) * np.linalg.norm(orientation))
        ratio = clamp(ratio, -1, 1)
        angle = np.degrees(np.arccos(ratio))
        sign = np.sign(np.cross(orientation, direction))
        angle *= sign
        return angle * (Ram.MAX_TURN / 180.0), 1-(np.sign(angle) * (angle) * (Ram.MAX_SPEED / 180.0))

    ''' main method for the ram ram algorithm that turns to face the enemy and charge towards it '''
    def ram_ram(self, bots: dict[str, any] = None):
        if not (bots and bots["huey"]):
            if self.huey_previous_positions:self.huey_previous_positions.append(self.huey_previous_positions[-1])
            print("🦐 Prev pos appended. 🦐")
            if self.huey_previous_orientations: self.huey_previous_orientations.append(self.huey_previous_orientations[-1])
            print("🦐 Prev orient appended. 🦐")

            # recovery!
            if (self.check_previous_position_and_orientation()):
                print("👿Start recovery👿")
                self.recovery_sequence()
                return self.huey_move(self.recover_speed, self.recover_turn)
            else:
                self.recovery_step = 0
                return self.huey_move(self.huey_old_speed, self.huey_old_turn)
         
        self.huey_old_position = self.huey_position if self.huey_position is not None else self.huey_old_position

        print("6, 7🫴🤪🫴")
        print("6 🤷‍♀️ 7")

        # if we don't have any data at all about orientation
        if self.huey_orientation is None:
            self.huey_orientation = self.huey_previous_orientations[-1]

        if self.huey_pos_count % 2 == 0: # Check every other frame for stationary
            print("ram ram 1: ")
            print("huey_position: " + str(self.huey_position))
            print("huey_orientation: " + str(self.huey_orientation))
            self.huey_previous_positions.append(self.huey_position) #TODO: null check?
            self.huey_previous_orientations.append(self.huey_orientation)

        self.huey_pos_count, self.huey_orient_count = self.huey_pos_count + 1, self.huey_orient_count + 1

        # Save Huey's last 10 positions & orientations
        if len(self.huey_previous_positions) > Ram.HISTORY_BUFFER:
            self.huey_previous_positions.pop(0)

        if len(self.huey_previous_orientations) > Ram.HISTORY_BUFFER:
            self.huey_previous_orientations.pop(0)
          
        # If the array for enemy_previous_positions is full, then pop the first one
        print("enemy_position: " + str(self.enemy_position))
        self.enemy_previous_positions.append(self.enemy_position)

        if len(self.enemy_previous_positions) > Ram.HISTORY_BUFFER:
            self.enemy_previous_positions.pop(0)

        if time.time() < self.recovering_until:
            print("🦋🌝Recovering...🌝🦋")
            return self.huey_move(self.recover_speed, self.recover_turn)
        else:
            self.recovering_until = 0
        
        # Check if Huey is stationary / unfound, recover if so
        if (self.check_previous_position_and_orientation()):
            print("👿Start recovery👿")
            self.recovery_sequence()
            return self.huey_move(self.recover_speed, self.recover_turn)
        else:
            self.recovery_step = 0

        # Get new position and heading values
        self.huey_position = np.array(bots['huey']['center']) if bots['huey']['center'] is not None else np.array(self.huey_old_position)
        self.huey_orientation = float(bots['huey']['orientation'] if bots['huey']['orientation'] is not None else self.huey_previous_orientations[-1])
    
        self.delta_t = time.time() - self.old_time  # record delta time
        self.old_time = time.time()
        
        if bots["enemy"]:
            self.enemy_position = np.array(bots['enemy']['center']) # probably issue here? 

            turn, speed = self.predict_desired_turn_and_speed()

            self.huey_old_turn, self.huey_old_speed = turn, speed
        
            # PID Shenanigans. Only use PID for the turn values
            if self.USE_PID and self.delta_t != 0:
                print("PID RAH")
                if self.delta_t > 0:
                    derivative = (self.huey_orientation - self.huey_previous_orientations[-1]) / (self.delta_t * 180.0)
                else:
                    derivative = 0
                
                pid_output = (turn * 1) + (derivative * 0.03 * -1)
                turn = clamp(pid_output, -1, 1)

            return self.huey_move(speed, turn)

        return self.huey_move(self.huey_old_speed, self.huey_old_turn) # TODO: can't see enemy, where do we go?
