import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .corner_detection_helpers import find_our_bot, find_centroids, compute_angle_between_midpoints, two_corners, math, deque, calc_diagonal_and_side_length

class RobotCornerDetection:
    """
    A class for detecting the corners and orientation of robots in images
    based on their unique colors and shapes.
    """
    
    LENGTH_BUFFER = 20

    def __init__(self, selected_colors: list, display_final_image: bool = False, display_possible_hueys: bool = False, color_percentage_rows = []):
        """
        Initializes the RobotCornerDetection class.

        Args:
            bots (list): A list of dictionary containing information about the 
                        bots, including bounding boxes and images. This is empty 
                        for now when initialized before the match. There is a
                        setter that will run during the match.
            selected_colors (list): Manually selected colors for front and back corners.
            display_final_image (bool): Whether to display the final image with
                        labeled left and right corners.
            display_possible_hueys (bool): Whether to display all possible
                        images of Huey.
        """
        self.bots = []
        self.selected_colors = selected_colors
        self.display_final_image = display_final_image
        self.display_possible_hueys = display_possible_hueys
        self.huey_color_percentage_threshold = -1
        self.color_percentage_rows = color_percentage_rows
        self.centroids = []
        self.diagonals = deque(maxlen=RobotCornerDetection.LENGTH_BUFFER)
        self.sides = deque(maxlen=RobotCornerDetection.LENGTH_BUFFER)
        self.left_diff = []
        self.right_diff = []
        self.ratio = []

        # plt.ion()
        # # fig, ax = plt.subplots()
        # # plt.plot(self.left_diff, label="Left Difference")
        # # plt.plot(self.right_diff, label="Right Difference")
        # plt.plot(self.diagonals, label = "Diagonals")
        # plt.plot(self.sides, label = "Sides")
        # plt.grid(True)
        # plt.legend()
        # plt.show()

    def set_bots(self, bots: dict):
        self.bots = bots
    
    def detect_our_robot_main(self, bot_images: list[np.ndarray], threshold_set=False) -> np.ndarray:
        """
        Detects the image containing our robot between two or more given images.

        Args:
            bot1_image (np.ndarray): The first bot image.
            bot2_image (np.ndarray): The second bot image.

        Returns:
            np.ndarray: The image identified as containing our robot.
        """
        try:
            if self.display_possible_hueys:
                window_width = 300
                window_height = 300

                for i, img in enumerate(bot_images):
                    if img is not None:
                        try:
                            resized_img = cv2.resize(img, (window_width, window_height))
                            cv2.imshow(f"Bot Image {i + 1}", resized_img)
                        except cv2.error as e:
                            print(f"Error resizing or displaying image {i + 1}: {e}")
                            continue
                    else:
                        print(f"Image {i + 1} is None")
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if bot_images and all(img is not None for img in bot_images):
                bot_color = self.selected_colors[0]
                our_bot = find_our_bot(self, bot_images, bot_color, threshold_set)

                return our_bot
            else:
                # print("No valid bot images found.")
                return None
        
        except Exception as e:
            print(f"Unexpected error in detect_our_robot_main: {e}")
            return None

    def corner_detection_main(self, previous_orientations: list = [], threshold_set: bool=False) -> dict | None:
        """
        Main function for detecting corners and orientation of the robot.

        Returns:
            dict: A dictionary containing details of the robot and enemy robots.
        """
        try:
            print("🚗")
            bot_images = [bot["img"] for bot in self.bots["bots"]]
            image = self.detect_our_robot_main(bot_images, threshold_set)
            
            if image is not None:
                print("👁️")
                # Find the identified bot (our robot)
                huey_bbox = None
                for bot_data in self.bots["bots"]:
                    if bot_data["img"] is image:
                        huey_bbox = bot_data["bbox"]
                        break
                print("🏇")

                huey = {
                    "bbox": huey_bbox,
                    "center": np.mean(huey_bbox, axis=0), # center of the bot with respect to the entire arena
                    "orientation": None,
                }
                print("🐺")

                # Enemy bots are all except the identified bot
                enemy_bots = {}
                if isinstance(self.bots, dict) and "bots" in self.bots:
                    for bot_data in self.bots["bots"]:
                        if bot_data["img"] is not image:
                            enemy_bots = {
                                "bbox": bot_data["bbox"],
                                "center": np.mean(bot_data["bbox"], axis=0),
                            }
                            break

                print("🐋")
                centroid_points = find_centroids(image, self.selected_colors)
                print("😬")
                self.centroids = centroid_points

                # print("pointy type", type(centroid_points[0][0]))

                print(self.centroids)
                
                # Every time we calculate 4 points, calculate diagonal and side length in the case of 1 front 1 back corner in the future
                calc_diagonal_and_side_length(self.centroids, self.diagonals, self.sides)
                        
                # print(f"🇬🇧🛌🇰🇷DIALGA: {self.diagonals}, 🇬🇧SYDNEY: {self.sides}")

                if (len(centroid_points[0]) + len(centroid_points[1]) == 2):
                    if previous_orientations is not None and len(previous_orientations) > 0:
                        previous_orientation = previous_orientations[0]
                        huey["orientation"] = two_corners(centroid_points, previous_orientation, self.diagonals, self.sides)
                    else:
                        huey["orientation"] = None
                    return {"huey": huey, "enemy": enemy_bots}

                elif (len(centroid_points[0]) + len(centroid_points[1]) < 2):
                    print("Less than 2 corners found")
                    return {"huey": huey, "enemy": enemy_bots}

                front_midpoint = (centroid_points[0][0] + centroid_points[0][1]) * 0.5
                back_midpoint = (centroid_points[1][0] + centroid_points[1][1]) * 0.5
                huey["orientation"] = compute_angle_between_midpoints(back_midpoint, front_midpoint)
                
                result = {"huey": huey, "enemy": enemy_bots}

                return result
            else:
                # print("Image doesn't exist")
                return {"huey": {}, "enemy": {}}

        except Exception as e:
            print(f"Unexpected error in corner_detection_main: {e}")
            return None


if __name__ == "__main__":
    huey_image_path = os.getcwd() + "/warped_images/east_4.png"
    not_huey_image_path = os.getcwd() + "/warped_images/east_4_not_huey.png"
    selected_colors_file = os.getcwd() + "/selected_colors.txt"
    
    try:
        huey_image = cv2.imread(huey_image_path)
        not_huey_image = cv2.imread(not_huey_image_path)
        
        if huey_image is None:
            raise ValueError(f"Failed to load image at path: {huey_image_path}")
        if not_huey_image is None:
            raise ValueError(f"Failed to load image at path: {not_huey_image_path}")
    
    except Exception as e:
        print(f"Error loading images: {e}")
        exit(1)

    housebot = {"bbox": [[0, 0], [1, 1]], "img": not_huey_image}
    bot1 = {"bbox": [[50, 50], [60, 60]], "img": not_huey_image}
    bot2 = {"bbox": [[150, 150], [160, 160]], "img": huey_image}
    bot3 = {"bbox": [[300, 150], [400, 180]], "img": not_huey_image}

    housebots = [housebot]
    bots = [bot1, bot2, bot3]
    all_bots = {"housebot": housebot, "bots": bots}
    
    selected_colors = []
    try:
        with open(selected_colors_file, "r") as file:
            for line in file:
                hsv = list(map(int, line.strip().split(", ")))
                selected_colors.append(hsv)
        if len(selected_colors) != 3:
            raise ValueError("The file must contain exactly 3 HSV values.")
    
    except Exception as e:
        print(f"Error reading selected_colors.txt: {e}")
        exit(1)
    
    corner_detection = RobotCornerDetection(selected_colors, True, False)
    corner_detection.set_bots(all_bots)
    result = corner_detection.corner_detection_main()
    print("result: " + str(result))
