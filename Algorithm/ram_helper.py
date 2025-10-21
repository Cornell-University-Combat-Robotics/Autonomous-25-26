import numpy as np
import time
import random
import math

BACK_UP_SPEED = -1
BACK_UP_TURN = 0
FORWARD_SPEED = 1
FORWARD_TURN = 0
LEFT_SPEED = 1
LEFT_TURN = -1
RIGHT_SPEED = 1
RIGHT_TURN = 1

def calculate_enemy_velocity(old_positions: list[np.array], curr_pos: np.array, dt: float):
    if len(old_positions) < 2 or dt == 0.0:
        return np.array([0.0, 0.0])

    # Start from the end and look for two consecutive valid positions
    i = len(old_positions) - 1
    while i > 0:
        if old_positions[i] is not None and old_positions[i-1] is not None:
            curr_pos = old_positions[i]
            old_pos = old_positions[i-1]
            return np.array(curr_pos) - np.array(old_pos)
        i -= 1
        
    return np.array([0.0, 0.0])

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
    flag = False
    if (predicted_position[0] > arena_width):
        predicted_position[0] = 1200
        flag = True
    if (predicted_position[0] < 0):
        predicted_position[0] = 0
        flag = True
    if (predicted_position[1] > arena_width):
        predicted_position[1] = 1200
        flag = True
    if (predicted_position[1] < 0):
        predicted_position[1] = 0
        flag = True
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

