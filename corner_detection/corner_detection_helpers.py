import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

FONT = cv2.FONT_HERSHEY_SIMPLEX
MIN_THRESHOLD = 0.08
CORNER_THRESHOLD = 10.0
BLACKOUT_THRESHOLD = 0.4

@staticmethod
def find_bot_color_pixels(image: np.ndarray, bot_color_hsv: list) -> int:
    """
    Detects the number of a predefined color pixels in the given image.

    Args: image (np.ndarray): Input image of the robot in BGR format.

    Returns: int: The number of predefined color pixels detected in the image.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the robot's color
    bot_color = np.array([bot_color_hsv[0], bot_color_hsv[1], bot_color_hsv[2]])

    # Create a mask for the robot's color in the image
    mask = cv2.inRange(hsv_image, bot_color, bot_color)

    # Count the number of non-zero pixels in the mask
    # cv2.imshow("Robot Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cv2.countNonZero(mask)

def get_contours_per_color(side: str, hsv_image: np.ndarray, selected_colors) -> list[np.ndarray]:
    """
    Retrieves contours for the front or back corners based on the manually picked color.

    Args:
        side (str): "front" for red contours, "back" for blue contours.
        hsv_image (np.ndarray): Input image in HSV format.

    Returns: list: Contours corresponding to the given color.
    """
    selected_color = (selected_colors[1] if side == "front" else selected_colors[2])

    # Define the HSV range around the selected color
    # We tried using 10 for the range; It was too large and picked up orange instead of red
    # For now, it is +-8
    selected_color_hsv = np.array([selected_color[0], selected_color[1], selected_color[2]])

    mask = cv2.inRange(hsv_image, selected_color_hsv, selected_color_hsv)

    # cv2.imshow("Corners Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_our_bot(self, images: list[np.ndarray], bot_color_hsv, threshold_set=True) -> np.ndarray | None:
    """
    Identifies which image contains our robot based on a predefined robot color.

    Args: images (list[np.ndarray]): List of input images.

    Returns: np.ndarray: The image containing our robot.
    """
    try:
        if not images:
            raise ValueError("The input image list is empty.")
        max_color_percentage = -1
        our_bot_image = None
        bot_color_percentages = []

        for image in images: # this for loop handles whether the image is the huey bot
            if image is None:
                print("Warning: One of the images is None, skipping...")
                continue
            
            color_pixel_count = find_bot_color_pixels(image, bot_color_hsv)
            image_area = image.size/3
            color_percentage = color_pixel_count/image_area
            bot_color_percentages.append(color_percentage)
                
            if color_percentage > max_color_percentage:
                our_bot_image = image
                max_color_percentage = color_percentage

        bot_color_percentages.sort()
        dynamic_threshold(self, threshold_set, bot_color_percentages)

        # Say we can't find huey if the bot's are below the huey threshold (for recovery)
        if len(bot_color_percentages) == 1 and bot_color_percentages[-1] < self.huey_color_percentage_threshold:
            our_bot_image = None
            # print("💔 check A")
        elif len(bot_color_percentages) > 1 and bot_color_percentages[-1] < min(MIN_THRESHOLD, self.huey_color_percentage_threshold):
            our_bot_image = None
            # print("💛 check B")
        
        return our_bot_image
    
    except Exception as e:
        print(f"Unexpected error occurred in find_our_bot: {e}")
        return None

def dynamic_threshold(self, threshold_set, bot_color_percentages):
    # Case 1: Setting the initial threshold 
    if not threshold_set:
        print("case 1")
        # Case 1.1: If we see 2 or more robots
        if len(bot_color_percentages) > 1 and bot_color_percentages[-1] > 0:
            huey_color_percentage = bot_color_percentages[-1]
            enemy_color_percentage = bot_color_percentages[-2]
            self.huey_color_percentage_threshold = max((huey_color_percentage + enemy_color_percentage) / 2, MIN_THRESHOLD)
            print("Initial Threshold (2+ robots): " + str(self.huey_color_percentage_threshold))
        # Case 1.2: If we see only 1 robot. We use the static threshold because that robot could be us or not
        else:
            self.huey_color_percentage_threshold = MIN_THRESHOLD
            print("Initial Threshold (1 robot): " + str(self.huey_color_percentage_threshold))
    
    # Case 2: Updating threshold and running sum using queue
    elif threshold_set and len(bot_color_percentages) >= 2:
        # Dynamic Threshold
        huey_color_percentage = bot_color_percentages[-1]
        enemy_color_percentage = bot_color_percentages[-2]

        if huey_color_percentage > 0:
            midpoint = (huey_color_percentage + enemy_color_percentage)/2
            if len(self.threshold_queue) == self.dynamic_threshold_window:
                # window is full so we can start caclulating
                popped = self.threshold_queue[0]
                self.running_sum += midpoint - popped
                # add the new midpoint and subtract the oldest
            else:
                self.running_sum += midpoint
                # add new midpoint if queue isnt full yet

            self.threshold_queue.append(midpoint)
            
            if len(self.threshold_queue) == self.dynamic_threshold_window:
                # set threshold if the queue is the size of the window we want to extract from
                self.huey_color_percentage_threshold = max(self.running_sum / self.dynamic_threshold_window, MIN_THRESHOLD)

    # Dynamic threshold and graphing data
    if len(bot_color_percentages) >= 2:
        self.color_percentage_rows.append((bot_color_percentages[-1], bot_color_percentages[-2],self.huey_color_percentage_threshold))
    if len(bot_color_percentages) == 1:
        if bot_color_percentages[0] > self.huey_color_percentage_threshold:
            self.color_percentage_rows.append((bot_color_percentages[0], 0,self.huey_color_percentage_threshold))
        else:
            self.color_percentage_rows.append((0, bot_color_percentages[0], self.huey_color_percentage_threshold))
    elif len(bot_color_percentages) == 0:
        self.color_percentage_rows.append((0, 0, self.huey_color_percentage_threshold))

def find_centroids_per_color(side: str, image: np.ndarray, hsv_image: np.ndarray, selected_colors, area_threshold) -> list:
    """
    Finds the centroids of a specific color (front or back) in the given image.

    Args:
        side (str): "front" or "back" for the color.
        image (np.ndarray): The input image in BGR format.
        hsv_image (np.ndarray): The HSV version of the input image.

    Returns: list: Centroids of the detected contours.
    """

    # 1. Get image dimensions and center point
    img_h, img_w = hsv_image.shape[:2]
    
    center_x, center_y = img_w // 2, img_h // 2

    # 2. Get contours from your helper function
    contours = get_contours_per_color(side, hsv_image, selected_colors)
    
    # Calculate bbox area:
    # 3. Define the sorting key (Distance is primary, Area is secondary)
    def sorting_criteria(c):
        area = cv2.contourArea(c)

        M = cv2.moments(c)
        if M["m00"] == 0:
            # Handle lines/points: assume the first point is the location
            # and push them to the end of the priority list
            return (float('inf'), 0)
        
        # Calculate centroid
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Euclidean distance squared from center
        dist_sq = (cx - center_x)**2 + (cy - center_y)**2
        
        # Sort by distance (ascending) then area (descending)
        return (dist_sq, -area)

    # 4. Sort the entire list
    sorted_contours = sorted(contours, key=sorting_criteria)
    sorted_contours = [c for c in sorted_contours if cv2.contourArea(c) >= CORNER_THRESHOLD]
    # print(f"🧏‍♂️ sorted areas: {[cv2.contourArea(c) for c in sorted_contours]}")
    # print(f"🧏‍♂️ sorted percentages: {[(cv2.contourArea(c)/(img_h * img_w)) for c in sorted_contours]}")
    # print(f"🚔🚔image area: {img_h * img_w}")

    # 5. Extract top 2 centroids
    centroids = []
    for contour in sorted_contours:
        area = cv2.contourArea(contour)
        print("Area", area)
        if area > area_threshold:
            if len(centroids) >= 2:
                break
                
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
            
    return centroids

def find_centroids(image: np.ndarray, selected_colors, area_threshold) -> np.ndarray:
    """
    Finds the centroids for the front and back corners of the robot.

    Args: image (np.ndarray): The input image in BGR format.

    Returns: list: A list containing centroids for the front and back corners.
    """
    three = False
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    centroid_front = find_centroids_per_color("front", image, hsv_image, selected_colors, area_threshold)
    centroid_back = find_centroids_per_color("back", image, hsv_image, selected_colors, area_threshold)
    num_corners = len(centroid_front) + len(centroid_back)
    #3 CORNERS LOGIC    
    # Check if we have incomplete points and use get_missing_point to fix it
    if len(centroid_front) == 1 and len(centroid_back) == 2:
        points = [centroid_front, centroid_back]
        centroid_front, centroid_back = get_missing_point(points)
        three = three or True
    elif len(centroid_back) == 1 and len(centroid_front) == 2:
        points = [centroid_front, centroid_back]
        centroid_front, centroid_back = get_missing_point(points)
        three = three or True

    # # Ensure we have exactly 2 points for front and back
    # if len(centroid_front) < 2 or len(centroid_back) < 2:
    #     return np.array([[], []])  # Return empty arrays if not enough points

    # Convert to numpy arrays with consistent shape
    front_array = np.array(centroid_front[:2])  # Take first 2 points if more exist
    back_array = np.array(centroid_back[:2])    # Take first 2 points if more exist

    return np.array([front_array, back_array], dtype=object), three

def two_corners(centroid_points: np.ndarray, previous_orientation: float, diagonals: list, sides: list, huey_bbox, prev_flipped:int, is_flipped: int) -> (float, bool):
    """
    Handles orientation calculation when only 2 points are detected by cases.
    Case 1: 2 Front or 2 Back corners are found
        - Finds the two possible angles
        - Finds the angle of the line between the center of Huey's bbox and the 
          midpoint of the 2 corners found.
        - Chooses the closer possible angle based on
          
    Case 2:
        Case 2.1: Same-Side Corners
        - Only one potential orientation: returns angle corr. front --> back
        Case 2.2: Diagonal Corners
        - Two potential orientations:
            - If valid orientation (i.e. not recently flipped), 
            return closest angle to prev  
            - Else return average of two potentials
    """
    IS_VALID_ORIE = prev_flipped == is_flipped
    print(f"IS_VALID_ORIE: {IS_VALID_ORIE}")
    print(f"🦭PREV ORIE: {previous_orientation}")

    # we update prev_flipped in corner_detection_main

    front_points = centroid_points[0]
    back_points = centroid_points[1]

    hx_min, hy_min, hx_max, hy_max = norm_from_bbox(huey_bbox)
    huey_center = ((hx_max - hx_min) / 2, (hy_max - hy_min) / 2)

    # CASE 1: Only 2 Front Corners detected OR Only 2 Back Corners detected
    if len(front_points) == 2 or len(back_points) == 2:
        
        points, frnt = (front_points, True) if len(front_points) == 2 else (back_points, False)
        
        point1, point2 = points[0], points[1]
        dx = point2[0] - point1[0]
        dy = -(point2[1] - point1[1]) # Flip Y for image coordinates
        
        line_angle = math.degrees(math.atan2(dy, dx))

        midpoint = [(a + b) / 2 for a, b in zip(point1, point2)]
        dy_math = -(midpoint[1] - huey_center[1])
        dx = (midpoint[0] - huey_center[0])
        direction_angle = (np.degrees(np.arctan2(dy_math, dx)) + 360.0) % 360.0
        if not frnt:
            direction_angle = (direction_angle + 180) % 360
        
        angle1 = (line_angle + 90) % 360 # Perpendicular possibilities
        angle2 = (line_angle - 90) % 360
        
        return pick_closest_angle(angle1, angle2, direction_angle), True

    # CASE 2: 1 Front and 1 Back Corner detected.
    elif len(front_points) == 1 and len(back_points) == 1:
        if len(diagonals) == 0:
            diagonal_avg = 93
            sides_avg = 67 # TODO: arbitrary default
        else:
            diagonal_avg = diagonals[0]
            sides_avg = sides[0]

        cutoff = (diagonal_avg + sides_avg)/2
        corner_distance = distance(front_points[0], back_points[0])
        dx = front_points[0][0] - back_points[0][0]
        dy = -(front_points[0][1] - back_points[0][1])
        angle = math.atan2(dy,dx) * (180/math.pi)
        
        # CASE 2.1: Both corners are on the same side
        if (corner_distance < cutoff):
            # print(f"🌫️🌫️🌫️🌫️🌫️CORNERS ON SAME SIDE: {angle} degrees")
            return angle, True
        
        # CASE 2.2: The corners are diagonal
        else:
            
            if not IS_VALID_ORIE: # take midorie
                print(f"💀MIDORIE")
                length = front_points[0][1] - back_points[0][1] # front[0][1] should be y coords,
                width = front_points[0][0] - back_points[0][0]
                hypotenuse = math.sqrt(math.pow(length, 2) + math.pow(width, 2))
                return math.asin(width/hypotenuse) * (180/math.pi), False
            
            else: 
                p1 = (angle + 45) % 360
                p2 = (angle - 45) % 360
                print(f"🌈🌈🌈CORNERS ON DIFFERENT SIDE: {p1} or {p2} degrees🌈🌈🌈")
                return pick_closest_angle(p1, p2, previous_orientation), False

    raise ValueError(f"Invalid point configuration: Front={len(front_points)}, Back={len(back_points)}")

def calc_diagonal_and_side_length(centroids, diagonal_len, side_len, len_nums):
    """
    Helper to calculate the diagonal and side length if we have 4 corners
    We use the average of the last 20 diagonal and side lenghts (1.5d and d) in the case of 1 front 1 back corner
    """
    
    if len(centroids) == 2 and len(centroids[0]) == len(centroids[1]) == 2:
        len_nums[0] += 1 
        # Left distances
        hypo_l = (np.linalg.norm(centroids[0][0] - centroids[1][1]))
        side_l = (np.linalg.norm(centroids[0][0] - centroids[1][0]))

        # Right distances
        hypo_r = (np.linalg.norm(centroids[0][1] - centroids[1][0]))
        side_r = (np.linalg.norm(centroids[0][1] - centroids[1][1]))

        if  hypo_l < side_l: # Identify longest as hypotenuse
            temp = hypo_l
            hypo_l = side_l
            side_l = temp
        
        if  hypo_r < side_r:
            temp = hypo_r
            hypo_r = side_r
            side_r = temp

        if len(diagonal_len) == 0 :
            diagonal_len.append((hypo_l + hypo_r)/2)
            side_len.append((side_l + side_r)/2)
        else:
            # take a waited average so that the average is resistent to changes
            diagonal_len[0] = (diagonal_len[0]*((len_nums[0]-1)/len_nums[0]) + hypo_l*((.5)/len_nums[0]) + hypo_r*((.5)/len_nums[0]))
            side_len[0] = (side_len[0]*((len_nums[0]-1)/len_nums[0]) + side_l*((.5)/len_nums[0]) + side_r*((.5)/len_nums[0]))

def pick_closest_angle(angle1: float, angle2: float, target: float) -> float:
    """Helper to find which candidate is closer to the previous orientation."""
    def get_diff(a, b):
        return abs((a - b + 180) % 360 - 180)


    return angle1 if get_diff(angle1, target) < get_diff(angle2, target) else angle2

def distance(point1: tuple, point2: tuple) -> float:
    """
    Calculates the Euclidean distance between two points.

    Args:
        point1 (tuple): The first point (x1, y1).
        point2 (tuple): The second point (x2, y2).

    Returns: float: The Euclidean distance.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
# THREE CORNERS
def get_missing_point(points: list) -> list:
    """
    Computes the missing point to form a complete set of red and blue points.

    Algorithm:
    - If given 2 blue points and 1 red point:
    1. Calculate the distance from each blue point to the red point.
    2. Identify the longer distance (hypotenuse).
    3. Copy the blue point associated with the hypotenuse near the red point
        to form the second red point.
    - If given 2 red points and 1 blue point:
    1. Calculate the distance from each red point to the blue point.
    2. Identify the longer distance (hypotenuse).
    3. Copy the red point associated with the hypotenuse near the blue point
        to form the second blue point.

    Args:
        points (list): A list containing two sublists:
            - points[0]: List of red points.
            - points[1]: List of blue points.

    Returns:
            list: A list containing updated red and blue points.
    """
    try:
        red_points = points[0]
        blue_points = points[1]

        if len(red_points) == 1 and len(blue_points) == 2:
            # Case #1: 1 red point and 2 blue points
            red_point = red_points[0]
            length_a = distance(blue_points[0], red_point)
            length_b = distance(blue_points[1], red_point)

            # Identify which blue point is associated with the hypotenuse
            if length_a > length_b:
                # Copy the blue point associated with length_a near the red point
                new_red_point = (
                    red_point[0] + (blue_points[0][0] - blue_points[1][0]),
                    red_point[1] + (blue_points[0][1] - blue_points[1][1]),
                )
                red_points.append((int(new_red_point[0]), int(new_red_point[1])))
            else:
                # Copy the blue point associated with length_b near the red point
                new_red_point = (
                    red_point[0] + (blue_points[1][0] - blue_points[0][0]),
                    red_point[1] + (blue_points[1][1] - blue_points[0][1]),
                )
                red_points.append((int(new_red_point[0]), int(new_red_point[1])))

        elif len(blue_points) == 1 and len(red_points) == 2:
            # Case #2: 2 red points and 1 blue point
            blue_point = blue_points[0]
            length_a = distance(red_points[0], blue_point)
            length_b = distance(red_points[1], blue_point)

            # Identify which red point is associated with the hypotenuse
            if length_a > length_b:
                # Copy the red point associated with length_a near the blue point
                new_blue_point = (
                    blue_point[0] + (red_points[0][0] - red_points[1][0]),
                    blue_point[1] + (red_points[0][1] - red_points[1][1]),
                )
                blue_points.append((int(new_blue_point[0]), int(new_blue_point[1])))
            else:
                # Copy the red point associated with length_b near the blue point
                new_blue_point = (
                    blue_point[0] + (red_points[1][0] - red_points[0][0]),
                    blue_point[1] + (red_points[1][1] - red_points[0][1]),
                )
                blue_points.append((int(new_blue_point[0]), int(new_blue_point[1])))

        return [red_points, blue_points]
    
    except Exception as e:
        print(f"Unexpected error in get_missing_point: {e}")
        return [[], []]

@staticmethod
def compute_tangent_angle(p1: tuple, p2: tuple) -> float: #NOTE: does not compute tangent angle anymore
    """
    Computes the angle of the tangent line to the front of the robot.

    Args:
        p1 (tuple): The first front point (x1, y1).
        p2 (tuple): The second front point (x2, y2).

    Returns: float: The angle of the tangent line relative to the x-axis in degrees.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = -(y2 - y1)
    angle_rad = np.arctan2(dy, dx)
    tangent_angle_rad = angle_rad + np.pi / 2
    return math.degrees(tangent_angle_rad) % 360

@staticmethod
def compute_angle_between_midpoints(p1: tuple, p2: tuple) -> float:
    """
    Computes the angle of the line between the front and back corners of robot.

    Args:
        p1 (tuple): The front midpoint (x1, y1).
        p2 (tuple): The back midpoint (x2, y2).

    Returns: float: The angle of the line between the points relative to the x-axis in degrees.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = -(y2 - y1)
    angle_rad = np.arctan2(dy, dx)
    return math.degrees(angle_rad) % 360

def display_image(image: np.ndarray, left_front: list, right_front: list):
    left_x, left_y = int(left_front[0]), int(left_front[1])
    right_x, right_y = int(right_front[0]), int(right_front[1])

    # Draw the left front corner
    cv2.circle(image, left_x, left_y, 5, (255, 255, 255), -1,)
    cv2.putText(image, "Left Front", left_x, left_y - 30, FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw the right front corner
    cv2.circle(image, right_x, right_y, 5, (255, 255, 255), -1)
    cv2.putText(image, "Right Front", right_x, right_y, - 30, FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Display the image
    cv2.imshow("Image with Left and Right Front Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def norm_from_bbox(bbox):
    """
    Gets four corner points from a bbox dictionary to reduce code reuse.
    """
    (x1, y1), (x2, y2) = bbox
    xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
    ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
    return xmin, ymin, xmax, ymax

def is_overlap(huey_bbox, enemy_bbox):
    """
    Checks whether the two bboxes interlap.

    Returns: Whether the two bboxes overlap
    """
    hx_min, hy_min, hx_max, hy_max = norm_from_bbox(huey_bbox)
    ex_min, ey_min, ex_max, ey_max = norm_from_bbox(enemy_bbox)
    return (hx_min < ex_max) and (hx_max > ex_min) and (hy_min < ey_max) and (hy_max > ey_min)

def compute_blackout_box(image, huey_bbox, enemy_bbox, thresh = BLACKOUT_THRESHOLD):
    """
    Blacks out intersection of huey and enemy_bbox onto the huey image. 
    If the interseciton is more than the threshold percent of huey's image,
    then we just return the original image.

    Args: image: should be hueys cropped image

    Returns: hueys image blacked out on the intersection if it was a small enough area
    """
    if image is None:
        return None, False
    # Huey x and y min and max
    hx_min, hy_min, hx_max, hy_max = norm_from_bbox(huey_bbox)
    # Enemy x and y min and max
    ex_min, ey_min, ex_max, ey_max = norm_from_bbox(enemy_bbox)

    # Intersection
    x_left = max(hx_min, ex_min)
    y_top = max(hy_min, ey_min)
    x_right = min(hx_max, ex_max)
    y_bottom = min(hy_max, ey_max)

    width = x_right - x_left
    height = y_bottom - y_top
    
    int_area = abs(width * height)
    huey_area = (hx_max - hx_min) * (hy_max - hy_min)
    
    # Check on whether we should blackout at all
    if int_area > huey_area * thresh:
        print("BLACKOUT IS HIGHER THAN 40%")
        return image, True

    # Arena to bbox cords
    xmin = int(round(x_left  - hx_min))
    xmax = int(round(x_right - hx_min))
    ymin = int(round(y_top   - hy_min))
    ymax = int(round(y_bottom  - hy_min))

    H, W = image.shape[:2]
    # Make sure borders aren't outside of image
    xmin = max(0, min(W, xmin))
    xmax = max(0, min(W, xmax))
    ymin = max(0, min(H, ymin))
    ymax = max(0, min(H, ymax))

    if xmin >= xmax or ymin >= ymax:
        return image, False

    # I think numpy indexing is (y,x)
    image[ymin:ymax, xmin:xmax] = np.asarray([255,255,255])
    return image, False

