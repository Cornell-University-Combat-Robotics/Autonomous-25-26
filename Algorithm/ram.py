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

        # THIS WORKS IN TESTBOX
        # self.turn_pid = PIDController(kp=0.008, ki=0.000, kd=0.0005, output_limits=(-1.0, 1.0))
        # self.speed_pid = PIDController(kp=0.003, ki=0.000, kd=0.000, output_limits=(-1.0,1.0))
        
        self.turn_pid = PIDController(kp=0.008, ki=0.000, kd=0.0005, output_limits=(-1.0, 1.0))
        self.speed_pid = PIDController(kp=0.003, ki=0.000, kd=0.000, output_limits=(-1.0,1.0))

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

        if counter_pos >= self.BACK_UP_THRESHOLD:
            self.is_recovering = True
            self.is_backing = False
            return True
        self.is_recovering = False
        self.is_backing = False
        return False
    
    def check_arena_edge(self, can_recover: bool = True):
        if not can_recover:
            self.is_recovering=False
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

    def predict_desired_angle_and_distance(self):
        ''' 
        Predict the signed heading error (in degrees) from Huey to the enemy target point,
        along with the Euclidean distance to that target point (in pixels).
        
        Coordinate conventions:
        - `self.huey_orientation` is interpreted in "math coords" (x right, y up).
        - Vision positions are stored in image coords (y down), so we convert positions to
            math coords using `invert_y(...)` before computing angle/distance.
        - Returned `angle_deg` is signed and continuous in [-180, 180]:
            * positive => counterclockwise turn needed
            * negative => clockwise turn needed

        Safety/edge cases:
        - If Huey is within `Ram.DANGER_ZONE` of the enemy, we force targeting to the enemy center.
        - If Huey and the target point coincide (or the direction vector is ~0), returns (0, 0).

        Preconditions:
        - `self.huey_position`, `self.enemy_position` are valid (x, y) numpy arrays.
        - `self.enemy_future_position` should be kept up-to-date by `get_enemy_orientation(...)`
            when using targeting methods 2 or 3.
        '''

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
        
        theta = np.radians(self.huey_orientation)
        forward = np.array([math.cos(theta), math.sin(theta)])  # math coords

        ef = invert_y(enemy_future_position)
        hp = invert_y(huey_position_copy)
        direction = ef - hp
        
        dist_px = float(np.linalg.norm(direction))
        if dist_px < 1e-6:
            return (0, 0)

        dot = float(np.dot(forward, direction))
        cross = float(forward[0] * direction[1] - forward[1] * direction[0])

        angle_deg = math.degrees(math.atan2(cross, dot))  # [-180, 180]
        return angle_deg, dist_px
    
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
    
    
    def get_enemy_orientation(self, bots):
        """
        Estimate the enemy's heading from its recent motion and update `self.enemy_future_position`
        to a target point on/near the enemy bbox.

        Coordinate conventions:
        - Positions (`self.enemy_position`, history) are in IMAGE coords (x right, y down).
        - Orientation is computed in "math-like degrees" using atan2(-dy_img, dx) and normalized to [0, 360).
        - The bbox projection uses `bbox_intersection()`, which expects the same IMAGE-coord convention.

        Fallbacks / stability:
        - If enemy/bbox is missing, or we have no prior positions, or we see invalid sentinel values
            ([-1, -1]), returns `self.enemy_old_orientation` without changing orientation.
        - If the frame-to-frame displacement is very small (`dist <= 2`), we treat the orientation as
            unreliable and return the last known good orientation. We use a small velocity lookahead to
            update `self.enemy_future_position` but still return `last_good`.

        Side effects:
        - Always resets `self.enemy_future_position` to the current enemy center at the start.
        - Updates `self.enemy_future_position` to a projected bbox edge point (or a velocity lookahead)
            when possible.
        - Updates `self.enemy_old_orientation` when a new reliable orientation is computed.

        Preconditions:
        - `self.enemy_position` has been updated for the current frame before calling this method.
        - `self.enemy_previous_positions` contains the previous frame's position (or more history).
        - `bots["enemy"]["bbox"]` is a pair of corners: [(x1,y1), (x2,y2)].

        Returns:
        float: enemy orientation in degrees in [0, 360), or the last known good orientation on fallback.
        """
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
            orientation, bots["enemy"]["bbox"], front = front)

        # TODO: Could add a weight like 0.8 or 1.2 times t for targeting
        self.enemy_future_position = cur_pos + t * forward_img
        self.enemy_old_orientation = orientation
        return orientation


    ''' main method for the ram ram algorithm that turns to face the enemy and charge towards it '''
    def ram_ram(self, bots: dict[str, any] = None, can_recover: bool = True, fps = 50, key=None):
        if self.is_recovering or self.is_backing:
            self.HISTORY_BUFFER = fps
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
            self.enemy_position = np.array(bots['enemy']['center'])
            error_angle, distance = self.predict_desired_angle_and_distance()
        
            if self.USE_PID and self.delta_t > 0:
                # 1. Calculate Turn using PID
                turn = self.turn_pid.update(error_angle, self.delta_t)
                
                # 2. Calculate Base Speed using PID
                ramming_distance = distance + 100 # 100 pixels is the "overshoot"
                base_speed = self.speed_pid.update(ramming_distance, self.delta_t)
                
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
            self.enemy_previous_positions.append(self.enemy_previous_positions[-1])
            self.enemy_position = self.enemy_previous_positions[-1]
            
            error_angle, distance = self.predict_desired_angle_and_distance()
            
            if self.USE_PID and self.delta_t > 0:
                turn = self.turn_pid.update(error_angle, self.delta_t)
                base_speed = self.speed_pid.update(distance + 100, self.delta_t)
                alignment_factor = math.cos(math.radians(clamp(error_angle, -90, 90))) ** 2
                speed = base_speed * alignment_factor
            else:
                turn = clamp(error_angle * (Ram.MAX_TURN / 180.0), -1, 1)
                speed = clamp(1 - (abs(error_angle) * (Ram.MAX_SPEED / 180.0)), -1, 1)
            
            self.huey_old_turn, self.huey_old_speed = turn, speed
            return self.huey_move(speed, turn)