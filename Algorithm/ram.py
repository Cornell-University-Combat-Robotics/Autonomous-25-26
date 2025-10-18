import math
import time
import numpy as np
import random
from .ram_helper import calculate_velocity, calculate_enemy_velocity, invert_y, check_wall
# import Algorithm.test_ram_csv as test_ram_csv


class Ram():
    # ----------------------------- CONSTANTS -----------------------------
    HISTORY_BUFFER = 10  # how many previous Huey or enemy position we are recording
    DANGER_ZONE = 55
    MAX_SPEED = 1  # magnitude between 0 and 1
    MAX_TURN = 1  # between 0 and 1
    ARENA_WIDTH = 1200  # in pixels
    TEST_MODE = False  # saves values to CSV file
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
    recovery_step = 0
    USE_PID = True

    def __init__(self, bots=None, huey_position=(np.array([ARENA_WIDTH, ARENA_WIDTH])), huey_old_position=(np.array([ARENA_WIDTH, ARENA_WIDTH])),
                 huey_orientation=45, enemy_position=np.array([0, 0]), huey_old_turn=0, huey_old_speed=0) -> None:
        # ----------------------------- INIT -----------------------------
        if bots is None:
            # initialize the position and orientation of huey
            self.huey_position = np.array(huey_position if huey_position is not None else (self.ARENA_WIDTH / 2, self.ARENA_WIDTH / 2), dtype=float)
            self.huey_old_position = np.array(huey_old_position if huey_old_position is not None else self.huey_position.copy(), dtype=float)
            # TODO: Fix orientation init
            self.huey_orientation = huey_orientation
            # initialize the current enemy position
            self.enemy_position = np.array(enemy_position if enemy_position is not None else (0.0, 0.0),dtype=float)
            self.huey_old_speed = huey_old_speed
            self.huey_old_turn = huey_old_turn
        else:
            self.huey_position = np.array(bots['huey'].get('center'))
            self.huey_old_position = np.array(bots['huey'].get('center'))
            self.huey_orientation = bots['huey'].get('orientation')
            self.enemy_position = np.array(bots['enemy'].get('center'))
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
        self.left, self.right = speed - turn, speed + turn
        if (self.left > 1):
            self.right -= self.left - 1
            self.left = 1
        if (self.right > 1):
            self.left -= self.right - 1
            self.right = 1
        return {'left': self.left, 'right': self.right, 'speed': speed, 'turn': turn}

    ''' 
    Returns the predicted desired orientation angle of the bot given all parameters, NOTE: the positive direction is counterclockwise
    Precondition: our_pos & enemy_position 
    '''
    def predict_desired_turn_and_speed(self, our_pos: np.array, our_orientation: float, enemy_pos: np.array, enemy_velocity: np.array, dt: float):
        # TODO: Think about using this method for all/some of the other variables
        i = 0
        while our_orientation == None and i < len(self.huey_previous_orientations):
            self.huey_orientation = our_orientation = self.huey_previous_orientations[-1-i]
            i -= 1
        
        # if we don't have any data at all about orientation
        if our_orientation == None:
            self.huey_orientation = our_orientation = 0
            self.huey_previous_orientations.append(our_orientation)
        
        check_wall(self.TEST_MODE, enemy_pos, 729)
        enemy_future_position = enemy_pos
        
        our_pos2 = np.copy(our_pos)
        if np.linalg.norm(enemy_pos - our_pos2) < Ram.DANGER_ZONE:
            enemy_future_position = enemy_pos
            if np.array_equal(enemy_pos, our_pos2):
                return (0, 0)
        
        if (np.array_equal(our_pos2, enemy_future_position)):
            return (0, 0)
        
        # return the angle in degrees
        our_orientation2 = np.radians(our_orientation)
        orientation = np.array(
            [math.cos(our_orientation2), math.sin(our_orientation2)])
        enemy_future_position2 = invert_y(enemy_future_position)
        our_pos3 = invert_y(our_pos2)
        direction = enemy_future_position2 - our_pos3
        
        # calculate the angle between the bot and the enemy
        ratio = np.dot(direction, orientation) / \
            (np.linalg.norm(direction) * np.linalg.norm(orientation))
        ratio = max(-1, min(1, ratio))
        angle = np.degrees(np.arccos(ratio))
        sign = np.sign(np.cross(orientation, direction))
        angle *= sign
        return angle * (Ram.MAX_TURN / 180.0), 1-(np.sign(angle) * (angle) * (Ram.MAX_SPEED / 180.0))
    
    ''' moves Huey backwards, forward, left, right'''
    def recovery_sequence(self):
        self.recovery_step += 1 
        print("RECOVERY STEP: ", self.recovery_step)
        duration = random.uniform(0.5, 1.0)
        self.recovering_until = time.time() + duration
        self.recover_speed = self.RECOVERY_SPEED_VALUES[self.recovery_step%4]
        self.recover_turn = self.RECOVERY_TURN_VALUES[self.recovery_step%4]

    def check_previous_position_and_orientation(self):
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
            return True
        
        return False

    ''' main method for the ram ram algorithm that turns to face the enemy and charge towards it '''
    def ram_ram(self, bots: dict[str, any] = None):
        self.huey_old_position = self.huey_position

        if self.huey_pos_count % 2 == 0: # Check every other frame for stationary
            self.huey_previous_positions.append(self.huey_position)
            self.huey_previous_orientations.append(self.huey_orientation)
        
        self.huey_pos_count, self.huey_orient_count = self.huey_pos_count + 1, self.huey_orient_count + 1

        # Save Huey's last 10 positions
        if len(self.huey_previous_positions) > Ram.HISTORY_BUFFER:
            self.huey_previous_positions.pop(0)

        if len(self.huey_previous_orientations) > Ram.HISTORY_BUFFER:
            self.huey_previous_orientations.pop(0)
          
        # If the array for enemy_previous_positions is full, then pop the first one
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
            if bots and bots["huey"]:
                self.huey_position = np.array(bots['huey'].get('center'))
                self.huey_previous_positions.append(self.huey_position)

                self.huey_orientation = bots['huey'].get('orientation')
                self.huey_previous_orientations.append(self.huey_orientation) 
                #TODO: add prev pos for when no bots seen? (since the last else is never reached)
            print("👿Start recovery👿")
            self.recovery_sequence()
            return self.huey_move(self.recover_speed, self.recover_turn)
        else:
            self.recovery_step = 0
        
        if bots and bots["huey"]: # TODO: address len > 0? 
            # Get new position and heading values
            self.huey_position = np.array(bots['huey'].get('center'))
            self.huey_orientation = bots['huey'].get('orientation')
        
            self.delta_t = time.time() - self.old_time  # record delta time
            self.old_time = time.time()
            
            if bots["enemy"]: # TODO: address len > 0? 
                self.enemy_position = np.array(bots['enemy'].get('center')) # probably issue here? 
            
                enemy_velocity = calculate_enemy_velocity(
                    self.enemy_previous_positions, self.enemy_position, self.delta_t)
                turn, speed = self.predict_desired_turn_and_speed(our_pos=self.huey_position, our_orientation=self.huey_orientation, enemy_pos=self.enemy_position,
                                                            enemy_velocity=enemy_velocity, dt=self.delta_t)
                
                self.huey_old_turn, self.huey_old_speed = turn, speed

                if (Ram.TEST_MODE):
                    angle = self.predict_desired_orientation_angle(
                        self.huey_position, self.huey_orientation, self.enemy_position, enemy_velocity, self.delta_t)
                    direction = self.predict_enemy_position(
                        self.enemy_position, enemy_velocity, self.delta_t) - self.huey_position
                    test_ram_csv.test_file_update(delta_time=self.delta_t, bots=bots, huey_pos=self.huey_position, huey_facing=self.huey_orientation,
                                                enemy_pos=self.enemy_position, huey_old_pos=self.huey_old_position,
                                                huey_velocity=calculate_velocity(
                                                    self.huey_position, self.huey_old_position, self.delta_t),
                                                enemy_old_pos=self.enemy_previous_positions, enemy_velocity=enemy_velocity, speed=speed, turn=turn,
                                                left_speed=self.left, right_speed=self.right, angle=angle, direction=direction)
            
                # PID Shenanigans. Only use PID for the turn values
                if self.USE_PID and self.delta_t != 0:
                    if self.delta_t > 0:
                        derivative = (self.huey_orientation - self.huey_previous_orientations[-1]) / (self.delta_t * 180.0)
                    else:
                        derivative = 0
                    pid_output = (turn * 1) + (derivative * 0.03 * -1)
                    turn = max(-1, min(1, pid_output))
                 
                return self.huey_move(speed, turn)
            return self.huey_move(self.huey_old_speed, self.huey_old_turn) # TODO: can't see enemy, where do we go?
        else:
            self.huey_previous_positions.append(self.huey_previous_positions[-1])
            print("🦐 Prev pos appended. 🦐")
            return self.huey_move(self.huey_old_speed, self.huey_old_turn)