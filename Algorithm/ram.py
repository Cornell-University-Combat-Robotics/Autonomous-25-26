import math
import random
import time
import cv2
from line_profiler import profile

import numpy as np

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
    DANGER_ZONE = 55 # TODO: for smarter algo
    MAX_SPEED = 1  # magnitude between 0 and 1
    MAX_TURN = 1  # between 0 and 1
    ARENA_WIDTH = 700  # in pixels
    TOLERANCE = 10  # how close Huey's prev pos are permitted to be
    BACK_UP_SPEED = -1
    BACK_UP_TURN = 0
    FORWARD_SPEED = 1
    FORWARD_TURN = 0
    LEFT_SPEED = 1
    LEFT_TURN = -1
    RIGHT_SPEED = 1
    RIGHT_TURN = 1
    BACK_UP_THRESHOLD = 10  # > Double EDGE_THRESHOLD
    EDGE_THRESHOLD = 5
    RECOVERY_SPEED_VALUES = [BACK_UP_SPEED* 0.8, FORWARD_SPEED* 0.8, LEFT_SPEED* 0.8, RIGHT_SPEED* 0.8] 
    RECOVERY_TURN_VALUES = [BACK_UP_TURN, FORWARD_TURN, LEFT_TURN, RIGHT_TURN]
    USE_PID = True
    is_recovering = False
    is_backing = False
    reverse = 0
    recovery_step = 0
    against_wall = ""
    moving_forward = -1

    def __init__(self, bots=None, huey_position=(np.array([ARENA_WIDTH, ARENA_WIDTH])), huey_old_position=(np.array([ARENA_WIDTH, ARENA_WIDTH])),
                 huey_orientation=45, enemy_position=np.array([0, 0]), enemy_orientation=0, huey_old_turn=0, huey_old_speed=0, is_recovering=False, targeting_method = 1) -> None:
        # ----------------------------- INIT -----------------------------
        if bots is None:
            # initialize the position and orientation of huey
            self.huey_position = np.array(huey_position if huey_position is not None else (self.ARENA_WIDTH / 2, self.ARENA_WIDTH / 2), dtype=float)
            self.huey_old_position = np.array(huey_old_position if huey_old_position is not None else self.huey_position.copy(), dtype=float)
            self.huey_orientation = float(huey_orientation if huey_orientation is not None else 0.0) # TODO: Fix orientation init
            self.huey_girth = 67
            
            # initialize the current enemy position
            self.enemy_position = np.array(enemy_position if enemy_position is not None else (0.0, 0.0), dtype=float)
            self.enemy_previous_positions = []
            self.enemy_previous_positions.append(self.enemy_position)
            self.enemy_future_position = self.enemy_position

            # initialize the current enemy orientation
            self.enemy_orientation = float(enemy_orientation if enemy_orientation is not None else 0.0)
            self.enemy_old_orientation = 0.0
            
        else:
            self.huey_position = init_values(bots, self.ARENA_WIDTH, is_pos=True, is_huey=True)
            self.huey_old_position = init_values(bots, self.ARENA_WIDTH, is_pos=True, is_huey=True)
            self.huey_orientation = init_values(bots, self.ARENA_WIDTH, is_pos=False, is_huey=True)
            self.enemy_position = init_values(bots, self.ARENA_WIDTH, is_pos=True, is_huey=False)
            self.enemy_future_position = self.enemy_position
            self.enemy_orientation = 0.0
            self.enemy_old_orientation = 0.0
            if bots["huey"] and len(bots["huey"]) > 0:
                self.huey_girth = (math.dist(bots['huey'].get('bbox')[1], bots['huey'].get('bbox')[0]))/2
            else:
                self.huey_girth = 67

        self.huey_old_speed = huey_old_speed
        self.huey_old_turn = huey_old_turn
        self.left = 0
        self.right = 0

        # initialize the huey position array
        self.huey_pos_count = 1
        self.huey_previous_positions = []
        self.huey_previous_positions.append(self.huey_position)

        # initialize the huey orientation array
        self.huey_orient_count = 1
        self.huey_previous_orientations = []
        self.huey_previous_orientations.append(self.huey_orientation)

        # initialize the enemy position array
        self.enemy_previous_positions = []
        self.enemy_previous_positions.append(self.enemy_position)
        self.enemy_future_position = self.enemy_position

        # # initialize the enemy orientation array
        # self.enemy_orient_count = 1
        # self.enemy_previous_orientations = []
        # self.enemy_previous_orientations.append(self.enemy_orientation)

        # old time
        self.old_time = time.time()
        # delta time
        self.delta_t = 0.001

        #recovery
        self.recovering_until = 2.0
        self.recover_speed = 0.5
        self.recover_turn = 0.5
        self.is_recovering = False
        self.is_backing = False
        # TODO: add this as an initializer
        self.targeting_method = targeting_method
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
        self.recovery_step += 1
        duration = random.uniform(0.5, 1.0)
        self.recovering_until = time.time() + duration
        self.recover_speed = self.RECOVERY_SPEED_VALUES[self.recovery_step%4]
        self.recover_turn = self.RECOVERY_TURN_VALUES[self.recovery_step%4]

    def check_previous_position_and_orientation(self, can_recover: bool = True):
        if not can_recover:
            self.is_recovering=False
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

        if counter_pos >= self.BACK_UP_THRESHOLD:
            self.is_recovering=True
            return True
        self.is_recovering=False
        return False
    
    def check_arena_edge(self, can_recover: bool = True):
        if not can_recover:
            self.is_recovering=False
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
    targeting method is where we set enemy_future position to. It defaults to self.enemy_position
    '''
    def predict_desired_turn_and_speed(self):
        check_wall(self.enemy_position)
        check_wall(self.enemy_future_position)
        assert self.targeting_method == 1 or self.targeting_method == 2 or self.targeting_method == 3
        match self.targeting_method:
            case 1:
                # center of bbox
                enemy_future_position = self.enemy_position
            case 2:
                # front of bbox
                enemy_future_position = self.enemy_future_position
            case 3:
                # back of bbox
                enemy_future_position = self.enemy_future_position
        
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
        
        # # CORRECT OLD CODE GO BACK TO THIS
        # # calculate the angle between the bot and the enemy
        # ratio = np.dot(direction, orientation) / \
        #     (np.linalg.norm(direction) * np.linalg.norm(orientation))
        # ratio = clamp(ratio, -1, 1)
        # angle = np.degrees(np.arccos(ratio))
        # sign = np.sign(np.cross(orientation, direction))
        # angle *= sign
        # return angle * (Ram.MAX_TURN / 180.0), 1-(np.sign(angle) * (angle) * (Ram.MAX_SPEED / 180.0))

        # SLOP: TESTIGN THIS CODE ITS OLD
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            return (0, 0)

        dot = float(np.dot(orientation, direction))
        cross = float(orientation[0] * direction[1] - orientation[1] * direction[0])

        angle_rad = math.atan2(cross, dot)                           # signed, continuous [-pi, pi]
        angle_deg = math.degrees(angle_rad)

        turn = clamp(angle_deg * Ram.MAX_TURN / 180.0, -1.0, 1.0)
        speed = 1.0 - min(1.0, abs(angle_deg) * Ram.MAX_TURN / 180.0)
        return turn, speed

    def bbox_intersection(self, orientation: float, bbox, front=1):
        """
        Given motion components in IMAGE coords (dx, dy_img) and an axis-aligned bbox (two corners),
        compute:
        - orientation: heading angle in degrees, normalized to [0, 360)
        - forward_img: unit forward vector in IMAGE coords (x right, y down)
        - t: scalar such that (center + t * forward_img) hits the bbox boundary

        bbox format assumed: bbox[0]=(x1,y1), bbox[1]=(x2,y2) (corners, any order)
        """
        assert (front == 1 or front == -1)
        # forward unit vector in IMAGE coords
        theta = np.radians(orientation)
        forward_img = np.array([np.cos(theta), -np.sin(theta)], dtype=float)

        # bbox half extents
        (x1, y1), (x2, y2) = bbox[0], bbox[1]
        half_w = 0.5 * abs(x2 - x1)
        half_h = 0.5 * abs(y2 - y1)

        # distance to first rectangle boundary along forward direction
        ux, uy = float(forward_img[0]), float(forward_img[1])
        if (abs(ux) != 0):
            tx = half_w / abs(ux)
        else:
            tx = float('inf')

        if (abs(uy) != 0):
            ty = half_h / abs(uy)
        else:
            ty = float('inf')

        t = min(tx, ty) * front

        return forward_img, t
    
    """
    Returns enemy orientation and updates self.enemy_future_position as well. Returns the last 
    known good orientation in a few cases: bots or bots[enemy] doesnt exits, no previous enemy positions, 
    or if it is a bad value. If the distance between the last two positions is low enough, then we revert
    to using the center of the bbox as enemy_future_position. 
    """
    def get_enemy_orientation(self, bots):
        if self.targeting_method == 1:
            front = 1
        elif self.targeting_method == 2:
            front = 1
        elif self.targeting_method == 3:
            front = -1

        self.enemy_future_position = self.enemy_position
        last_good = self.enemy_old_orientation
        # TODO: Think about adding better logic here.
        if not (bots and bots.get("enemy") and bots["enemy"].get("bbox") is not None):
            return last_good
        
        if len(self.enemy_previous_positions) == 0:
            return last_good
        
        prev_pos = self.enemy_previous_positions[-1]
        cur_pos = self.enemy_position

        if np.array_equal(cur_pos, np.array([-1.0, -1.0])) or np.array_equal(prev_pos, np.array([-1.0, -1.0])):
            return last_good
        
        delta_img = cur_pos - prev_pos
        dist = np.linalg.norm(delta_img)

        # This is what controls whether we use a side of a bbox or velocity based
        if dist <= 2:
            # If bbox missing, we can still do a velocity-based future estimate
            # SLOP: THIS IS PURELY TESTING CODE, AHJSDAKSJHJDLKASJd
            if len(self.enemy_previous_positions) > 0:
                prev = self.enemy_previous_positions[-1]
                cur = self.enemy_position
                if prev is not None and cur is not None:
                    v = cur - prev
                    if np.isfinite(v).all() and np.linalg.norm(v) < 250:  # reject teleports
                        # TODO: LOOK AT THIS 4, IDK
                        lookahead_frames = 4
                        next = cur + v * lookahead_frames * front
                        next = np.array(next, dtype=float)
                        check_wall(next)
                        self.enemy_future_position = next
            
            return last_good
        
        # The only thing that actuallu calculates orientation
        dx = float(delta_img[0])
        dy_img = float(delta_img[1])
        orientation = (np.degrees(np.arctan2(-dy_img, dx)) + 360.0) % 360.0

        forward_img, t = self.bbox_intersection(
            orientation, bots["enemy"]["bbox"], front = front
        )

        # TODO: Could add a weight like 0.8 or 1.2 times t for targeting
        self.enemy_future_position = cur_pos + t * forward_img
        self.enemy_old_orientation = orientation      
        if True:

            print(f"🇳🇱enemy possy🇳🇱: {self.enemy_position}")
            print(f"🏓ENEM FUT POS:🏓 {self.enemy_future_position}")
            print(f"dx💩 {dx}💩")
            print(f"dy💩 {dy_img}💩")
            print(f"❤️traj: {orientation}❤️")
        
        return orientation


    ''' main method for the ram ram algorithm that turns to face the enemy and charge towards it '''
    def ram_ram(self, bots: dict[str, any] = None, can_recover: bool = True, fps = 50, key=None):
        if self.is_recovering or self.is_backing:
            self.HISTORY_BUFFER = fps*5/2
        else:
            self.HISTORY_BUFFER = fps*5
        self.BACK_UP_THRESHOLD = 0.75*self.HISTORY_BUFFER
        self.EDGE_THRESHOLD = 0.25*self.HISTORY_BUFFER
        
        if key == ord("r"):  # Press Q on keyboard to exit
            print("Recovery key r pressed.")
            self.huey_previous_positions = []
            self.huey_previous_orientations = []
            self.huey_previous_positions.append(self.huey_position)
            self.huey_previous_orientations.append(self.huey_orientation)

        if self.huey_pos_count % 1 == 0:
            self.huey_previous_positions.append(self.huey_position)
            self.huey_previous_orientations.append(self.huey_orientation)

            # print(f'🥶🥶🥶 Huey Pos Count: {self.huey_pos_count}')
        self.huey_pos_count += 1
        self.huey_orient_count += 1

        # Save Huey's last 10 positions
        if len(self.huey_previous_positions) > self.HISTORY_BUFFER:
            self.huey_previous_positions = self.huey_previous_positions[int(len(self.huey_previous_positions)-self.HISTORY_BUFFER):]

        if len(self.huey_previous_orientations) > self.HISTORY_BUFFER:
            self.huey_previous_orientations = self.huey_previous_orientations[int(len(self.huey_previous_orientations)-self.HISTORY_BUFFER):]
        
        if bots and bots['enemy']:
            self.enemy_position = np.array(bots['enemy']['center'])
        
        self.enemy_orientation = self.get_enemy_orientation(bots)
        
        # If the array for enemy_previous_positions is full, then pop the first one
        self.enemy_previous_positions.append(self.enemy_position)
        if len(self.enemy_previous_positions) > self.HISTORY_BUFFER:
            self.enemy_previous_positions = self.enemy_previous_positions[-int(self.HISTORY_BUFFER):]

        if len(self.enemy_previous_positions) > self.HISTORY_BUFFER:
            self.enemy_previous_positions = self.enemy_previous_positions[int(len(self.enemy_previous_positions)-self.HISTORY_BUFFER):]
        
        if time.time() < self.recovering_until:
            # print("Recovering...")
            return self.huey_move(self.recover_speed, self.recover_turn)
        else:
            self.recovering_until = 0

        # print(f"💅POPOS:💅 {self.huey_position}")
        # print(f"🛸ORORIE:🛸 {self.huey_orientation}")
        # print(f"🦒🦒🦒GIRTH {self.huey_girth}")

        backup = self.check_arena_edge()
        if backup == 1:
            self.is_backing = True
            return self.huey_move(self.FORWARD_SPEED, self.FORWARD_TURN)
        elif backup == -1:
            self.is_backing = True
            return self.huey_move(self.BACK_UP_SPEED, self.BACK_UP_TURN)
        self.is_backing = False
            
        if (self.check_previous_position_and_orientation(can_recover)):
            if (bots and bots["huey"] and len(bots["huey"]) > 0):
                self.huey_position = np.array(bots['huey'].get('center'))
                self.huey_previous_positions.append(self.huey_position)

                if (bots["huey"].get("orientation") is not None):
                    self.huey_orientation = bots['huey'].get('orientation')
                    self.huey_previous_orientations.append(self.huey_orientation)
                else:
                    self.huey_previous_orientations.append(self.huey_previous_orientations[-1])
            else:
                self.huey_previous_positions.append(self.huey_previous_positions[-1])
            # print("Start 🍀SPORADIH🍀🍀🍀")
            self.recovery_sequence() #SEQUENCE
            return self.huey_move(self.recover_speed, self.recover_turn)
        else:
            self.recovery_step = 0
        
        if bots and bots["huey"] and len(bots["huey"])>0:
            self.huey_girth = (math.dist(bots['huey'].get('bbox')[1], bots['huey'].get('bbox')[0]))/2
            self.huey_position = np.array(bots['huey'].get('center'))
            if (bots["huey"].get("orientation") is not None):
                self.huey_orientation = bots['huey'].get('orientation')

            self.delta_t = time.perf_counter() - self.old_time
            self.old_time = time.perf_counter()
        else:
            self.huey_previous_positions.append(self.huey_previous_positions[-1])
            self.huey_previous_orientations.append(self.huey_previous_orientations[-1])
            # print("Prev pos appended.")
            return self.huey_move(self.huey_old_speed, self.huey_old_turn)

        if bots["enemy"]:
            # self.enemy_position = np.array(bots['enemy']['center']) # probably issue here? 
            turn, speed = self.predict_desired_turn_and_speed()
            self.huey_old_turn, self.huey_old_speed = turn, speed
        
            # PID Shenanigans. Only use PID for the turn values
            if self.USE_PID and self.delta_t != 0:
                if self.delta_t > 0:
                    d_orientation = ((self.huey_orientation - self.huey_previous_orientations[-1] + 180) % 360 ) - 180
                    d_time = self.delta_t * 180.0
                    derivative = d_orientation / d_time
                else:
                    derivative = 0
                
                pid_output = (turn * 0.8) + (derivative * 0.04 * -1)
                turn = clamp(pid_output, -1, 1)

            return self.huey_move(speed, turn)
        else:
            print("enemy bot not detected, previous position appended")
            self.enemy_position = self.enemy_previous_positions[-1]
            turn, speed = self.predict_desired_turn_and_speed()
            self.huey_old_turn, self.huey_old_speed = turn, speed
            return self.huey_move(speed, turn)
