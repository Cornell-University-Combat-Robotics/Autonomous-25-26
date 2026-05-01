import math
import cv2
import numpy as np

def draw_orientation_arrow(frame, detected_bots_with_data, arrow_length=50, thickness=2):
    for bot_name, data in detected_bots_with_data[0].items():
        if not data or 'orientation' not in data or 'bbox' not in data:
            continue

        x, y, w, h = data['bbox']
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        angle_rad = math.radians(data['orientation'])
        ex = int(cx + arrow_length * math.cos(angle_rad))
        ey = int(cy - arrow_length * math.sin(angle_rad)) 

        cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 255, 255), thickness, tipLength=0.3)
        cv2.putText(frame, f"{bot_name}: {data['orientation']:.1f}°",
                    (cx + 5, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
def draw_manual_arrow(frame, cx, cy, theta, arrow_length = 50, thickness = 2):
    angle_rad = math.radians(theta)
    ex = int(cx + arrow_length * math.cos(angle_rad))
    ey = int(cy - arrow_length * math.sin(angle_rad))
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), (255, 0, 255), thickness, tipLength=0.3)


def get_darkness_score(image, selected_colors):
    '''NOT IN USE -- BROKEN'''
    bot_color_rgb = np.array([selected_colors[0][0], selected_colors[0][1], selected_colors[0][2]])

    # bot_color_hsv = cv2.cvtColor(bot_color_rgb, cv2.COLOR_BGR2HSV)
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # mask = cv2.inRange(hsv_image, bot_color_hsv, bot_color_hsv)

    #TODO: Figure out masking? (display real version)

    mask = cv2.inRange(image, bot_color_rgb, bot_color_rgb)
    print(bot_color_rgb)

    # lower_green = np.array([0, 100, 0])
    # print(lower_green)
    # upper_green = np.array([100, 255, 100])
    # mask = cv2.inRange(image, lower_green, upper_green)

    cv2.imshow("Robot Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(cv2.countNonZero(mask))

def angle_difference(a1, a2):
    """
    Returns the smallest difference between two angles (in degrees).
    Result is always between 0 and 180.
    """
    diff = (a2 - a1) % 360
    if diff > 180:
        diff = 360 - diff
    return abs(diff)


