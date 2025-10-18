import numpy as np
import time
import numpy as np
import random

BACK_UP_SPEED = -1
BACK_UP_TURN = 0
FORWARD_SPEED = 1
FORWARD_TURN = 0
LEFT_SPEED = 1
LEFT_TURN = -1
RIGHT_SPEED = 1
RIGHT_TURN = 1
RECOVERY_SPEED_VALUES = [BACK_UP_SPEED, FORWARD_SPEED, LEFT_SPEED, RIGHT_SPEED]
RECOVERY_TURN_VALUES = [BACK_UP_TURN, FORWARD_TURN, LEFT_TURN, RIGHT_TURN]

''' 
calculate the velocity of the bot given the current and previous position
Precondition: dt >= 0, if dt == 0, the velocity == 0; curr_pos & old_pos: 2-value array [x,y] 
x & y: > 0 
'''
def calculate_velocity(old_pos: np.array, curr_pos: np.array, dt: float):
    if (dt == 0.0):
        return np.array([0.0, 0.0])
    return (curr_pos - old_pos)

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
def check_wall(test_mode, predicted_position: np.array, arena_width=1200):
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
    if (test_mode and flag):
        print("moved that jon")