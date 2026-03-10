import math
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

FONT = cv2.FONT_HERSHEY_SIMPLEX
MIN_THRESHOLD = 0.035
CORNER_THRESHOLD = 10.0

#GRAPH CODE
filename = "corner_percentage.csv"
areafile = "area.csv"

# create CSV file once with header
with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["corner_percentage"])

with open(areafile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["area"])
@staticmethod
def find_bot_color_pixels(image: np.ndarray, bot_color_hsv: list) -> int:
    """
    Detects the number of a predefined color pixels in the given image.

    Args:
        image (np.ndarray): Input image of the robot in BGR format.

    Returns:
        int: The number of predefined color pixels detected in the image.
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

    Returns:
        list: Contours corresponding to the given color.
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

def find_our_bot(self, images: list[np.ndarray], bot_color_hsv, threshold_set=False) -> np.ndarray | None:
    """
    Identifies which image contains our robot based on a predefined robot color.

    Args:
        images (list[np.ndarray]): List of input images.

    Returns:
        np.ndarray: The image containing our robot.
    """
    try:
        if not images:
            raise ValueError("The input image list is empty.")
        max_color_percentage = -1
        our_bot_image = None 

        bot_color_percentages = []
        max_bot_area = 0.0

        for image in images: # this for loop handles whether the image is the huey bot
            if image is None:
                print("Warning: One of the images is None, skipping...")
                continue
            
            color_pixel_count = find_bot_color_pixels(image, bot_color_hsv)
            # print(f"🎎bot colors {color_pixel_count}")

            image_area = image.size/3

            color_percentage = color_pixel_count/image_area

            bot_color_percentages.append(color_percentage)
            # check if the next if statement is redundant since we are tracking the percentages with the list...
            # and sorting it to find the max. potentially, we do not need this calculation. Another note,
            # the color percentages are very variable during the match so i don't know if we should just 
            # continuously set a max.      
            if color_percentage > max_color_percentage:
                our_bot_image = image
                max_color_percentage = color_percentage
                max_bot_area = color_pixel_count

        bot_color_percentages.sort()
        # self.color_percentage_rows.append((bot_color_percentages[-1], bot_color_percentages[-2]))
        
        if threshold_set: # Set initial threshold
            if len(bot_color_percentages) > 1 and bot_color_percentages[-1] > 0:
                self.huey_color_percentage_threshold = max((bot_color_percentages[-1] + bot_color_percentages[-2]) / 2, MIN_THRESHOLD)
                print("Threshold: " + str(self.huey_color_percentage_threshold))
            elif len(bot_color_percentages) == 1:
                self.huey_color_percentage_threshold = max(bot_color_percentages[0] - 0.075, MIN_THRESHOLD)
                print("Threshold: " + str(self.huey_color_percentage_threshold))
        if len(bot_color_percentages) == 1 and max_color_percentage < self.huey_color_percentage_threshold:
            our_bot_image = None
        elif len(bot_color_percentages) > 1 and max_color_percentage < min(MIN_THRESHOLD, self.huey_color_percentage_threshold):
            our_bot_image = None
        
        # Writing information to be graphed
        if len(bot_color_percentages) >= 2:
            self.color_percentage_rows.append((bot_color_percentages[-1], bot_color_percentages[-2],self.huey_color_percentage_threshold))
        if len(bot_color_percentages) == 1:
            if bot_color_percentages[0] > self.huey_color_percentage_threshold:
                self.color_percentage_rows.append((bot_color_percentages[0], 0,self.huey_color_percentage_threshold))
            else:
                self.color_percentage_rows.append((0, bot_color_percentages[0], self.huey_color_percentage_threshold))
        elif len(bot_color_percentages) == 0:
            self.color_percentage_rows.append((0, 0, self.huey_color_percentage_threshold))

        # if our_bot_image is None:
        #     print("Huey is not found")
            
        # cv2.imshow("OUR BOT!", our_bot_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return our_bot_image
    
    except Exception as e:
        print(f"Unexpected error occurred in find_our_bot: {e}")
        return None

def find_centroids_per_color(side: str, image: np.ndarray, hsv_image: np.ndarray, selected_colors) -> list:
    """
    Finds the centroids of a specific color (front or back) in the given image.

    Args:
        side (str): "front" or "back" for the color.
        image (np.ndarray): The input image in BGR format.
        hsv_image (np.ndarray): The HSV version of the input image.

    Returns:
        list: Centroids of the detected contours.
    """

    # 1. Get image dimensions and center point
    img_h, img_w = hsv_image.shape[:2]
    
    center_x, center_y = img_w // 2, img_h // 2

    # 2. Get contours from your helper function
    contours = get_contours_per_color(side, hsv_image, selected_colors)
    
    # Calculate bbox area:
    bbArea = img_h * img_w
    # 3. Define the sorting key (Distance is primary, Area is secondary)
    def sorting_criteria(c):
        area = cv2.contourArea(c)
        corner_percentage = area / (bbArea)
        # log area value
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([corner_percentage])

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
    print(f"🧏‍♂️ sorted areas: {[cv2.contourArea(c) for c in sorted_contours]}")
    print(f"🧏‍♂️ sorted percentages: {[(cv2.contourArea(c)/(image.size / 3)) for c in sorted_contours]}")

    # 5. Extract top 2 centroids
    centroids = []
    for contour in sorted_contours:
        if len(centroids) >= 2:
            break
            
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
            
    return centroids

def find_centroids(image: np.ndarray, selected_colors) -> np.ndarray:
    """
    Finds the centroids for the front and back corners of the robot.

    Args:
        image (np.ndarray): The input image in BGR format.

    Returns:
        list: A list containing centroids for the front and back corners.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    centroid_front = find_centroids_per_color("front", image, hsv_image, selected_colors)
    centroid_back = find_centroids_per_color("back", image, hsv_image, selected_colors)

    # Check if we have incomplete points and use get_missing_point to fix it
    if len(centroid_front) == 1 and len(centroid_back) == 2:
        points = [centroid_front, centroid_back]
        centroid_front, centroid_back = get_missing_point(points)
    elif len(centroid_back) == 1 and len(centroid_front) == 2:
        points = [centroid_front, centroid_back]
        centroid_front, centroid_back = get_missing_point(points)

    # # Ensure we have exactly 2 points for front and back
    # if len(centroid_front) < 2 or len(centroid_back) < 2:
    #     return np.array([[], []])  # Return empty arrays if not enough points

    # Convert to numpy arrays with consistent shape
    front_array = np.array(centroid_front[:2])  # Take first 2 points if more exist
    back_array = np.array(centroid_back[:2])    # Take first 2 points if more exist

    return np.array([front_array, back_array], dtype=object)

def two_corners(centroid_points: np.ndarray, previous_orientation: float, diagonals: list, sides: list) -> float:
    """
    Handles orientation calculation when only 2 points are detected.
    """
    print("🐼")
    front_points = centroid_points[0]
    back_points = centroid_points[1]

    # CASE 1: Only 2 Front Corners detected OR Only 2 Back Corners detected
    print("🐒")
    if len(front_points) == 2 or len(back_points) == 2:
        # Correctly pick the points based on which list has 2
        points = front_points if len(front_points) == 2 else back_points
        print("🙊🙊🙊")
        point1, point2 = points[0], points[1]
        dx = point2[0] - point1[0]
        dy = -(point2[1] - point1[1]) # Flip Y for image coordinates
        print("🙈🙈🙈")
        
        line_angle = math.degrees(math.atan2(dy, dx))

        angle1 = (line_angle + 90) % 360 # Perpendicular possibilities
        angle2 = (line_angle - 90) % 360

        
        
        return pick_closest_angle(angle1, angle2, previous_orientation)

    # CASE 2: 1 Front and 1 Back Corner detected.
    elif len(front_points) == 1 and len(back_points) == 1:
        if len(diagonals) > 0:
            print("🦧")
            diagonal_avg = diagonals[0]
            sides_avg = sides[0]
            cutoff = (diagonal_avg + sides_avg)/2
            corner_distance = distance(front_points[0], back_points[0])
            print("💛💛💛")
            dx = front_points[0][0] - back_points[0][0]
            dy = -(front_points[0][1] - back_points[0][1])
            print("💛")
            angle = math.atan2(dy,dx) * (180/math.pi)
            
            # CASE 2.1: Both corners are on the same side
            if (corner_distance < cutoff):
                print(f"🌫️🌫️🌫️🌫️🌫️CORNERS ON SAME SIDE: {angle} degrees")
                return angle
            
            # CASE 2.2: The corners are diagonal
            else:
                p1 = (angle + 45) % 360
                p2 = (angle - 45) % 360
                print(f"🌈🌈🌈CORNERS ON DIFFERENT SIDE: {p1} or {p2} degrees🌈🌈🌈")
                
                return pick_closest_angle(p1, p2, previous_orientation)

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

    Returns:
        float: The Euclidean distance.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

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

    Returns:
        float: The angle of the tangent line relative to the x-axis in degrees.
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

    Returns:
        float: The angle of the line between the points relative to the x-axis in degrees.
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
