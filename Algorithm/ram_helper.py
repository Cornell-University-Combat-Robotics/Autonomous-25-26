import numpy as np
import math

BACK_UP_SPEED = -1
BACK_UP_TURN = 0
FORWARD_SPEED = 1
FORWARD_TURN = 0
LEFT_SPEED = 1
LEFT_TURN = -1
RIGHT_SPEED = 1
RIGHT_TURN = 1

'''
inverting the y position
Precondition: np.array
'''
def invert_y(pos: np.array):
    pos2 = np.copy(pos)
    pos2[1] = -pos[1]
    return pos2

""" if enemy robot predicted position is outside of arena, move it inside. """
def check_wall(predicted_position: np.array, arena_width=1200):
    if (predicted_position[0] > arena_width):
        predicted_position[0] = 1200
    if (predicted_position[0] < 0):
        predicted_position[0] = 0
    if (predicted_position[1] > arena_width):
        predicted_position[1] = 1200
    if (predicted_position[1] < 0):
        predicted_position[1] = 0
    print("moved that jon")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def to_float(x, fallback=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float(fallback)
        return v
    except Exception:
        return float(fallback)
    
def mix_speed_turn(speed, turn):
    # symmetric, safe motor mixing
    left  = clamp(speed - turn, -1, 1)
    right = clamp(speed + turn, -1, 1)
    return left, right

# self.huey_position = np.array(bots['huey']['center'] if np.array(bots['huey']['center']) is not None else (self.ARENA_WIDTH / 2, self.ARENA_WIDTH / 2), dtype=float)

def initialize_values(bots: list, arena_width: int, is_pos: bool, is_enemy: bool, enemy_position: np.array):
    if (is_enemy): #TODO: note done
        if (is_pos):
            value = np.array(enemy_position if enemy_position is not None else (0.0, 0.0), dtype=float)
        else:
            value = np.array(bots['enemy']['center'] if bots['enemy']['center'] is not None else (0.0, 0.0), dtype=float)
    else:
        if (is_pos): # Huey Position
            value = np.array(bots['huey']['center'] if np.array(bots['huey']['center']) is not None else (arena_width / 2, arena_width / 2), dtype=float)
        else:       # Orientation
            value = float(bots['huey']['orientation'] if bots['huey']['orientation'] is not None else 0.0)
    
    return value
    