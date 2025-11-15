import os
import cv2
import numpy as np
from .corner_detection_helpers import find_our_bot, find_centroids, compute_angle_between_midpoints, get_left_and_right_front_points, display_image

class RobotCornerDetection:
    """
    A class for detecting the corners and orientation of robots in images
    based on their unique colors and shapes.
    """
    
    def __init__(self, selected_colors: list, display_final_image: bool = False, display_possible_hueys: bool = False):
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

    def set_bots(self, bots: dict):
        self.bots = bots
    
    def detect_our_robot_main(self, bot_images: list[np.ndarray]) -> np.ndarray:
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
                our_bot = find_our_bot(bot_images, bot_color)

                return our_bot
            else:
                print("No valid bot images found.")
                return None
        
        except Exception as e:
            print(f"Unexpected error in detect_our_robot_main: {e}")
            return None

    def corner_detection_main(self) -> dict | None:
        """
        Main function for detecting corners and orientation of the robot.

        Returns:
            dict: A dictionary containing details of the robot and enemy robots.
        """
        try:
            bot_images = [bot["img"] for bot in self.bots["bots"]]
            image = self.detect_our_robot_main(bot_images)
            
            if image is not None:
                centroid_points = find_centroids(image, self.selected_colors)

                # For displaying centroids
                left_front, right_front = get_left_and_right_front_points(centroid_points)

                if left_front is None or right_front is None:
                    print("Could not determine left/right front points")
                    # return {"huey": {}, "enemy": {}}
                    orientation = 0.0
                else:
                    # Calculate orientation
                    front_midpoint = (centroid_points[0][0] + centroid_points[0][1]) * 0.5

                    back_midpoint = (centroid_points[1][0] + centroid_points[1][1]) * 0.5
                    
                    orientation = compute_angle_between_midpoints(back_midpoint, front_midpoint)

                # Find the identified bot (our robot)
                huey_bbox = None
                for bot_data in self.bots["bots"]:
                    if bot_data["img"] is image:
                        huey_bbox = bot_data["bbox"]
                        break

                huey = {
                    "bbox": huey_bbox,
                    "center": np.mean(huey_bbox, axis=0),
                    "orientation": orientation,
                }

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
                
                result = {"huey": huey, "enemy": enemy_bots}
                if self.display_final_image:
                    display_image()
                
                if self.display_final_image:
                    # Draw the left front corner
                    cv2.circle(
                        image,
                        (int(left_front[0]), int(left_front[1])),
                        5,
                        (255, 255, 255),
                        -1,
                    )
                    cv2.putText(
                        image,
                        "Left Front",
                        (int(left_front[0]), int(left_front[1]) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                    # Draw the right front corner
                    cv2.circle(
                        image,
                        (int(right_front[0]), int(right_front[1])),
                        5,
                        (255, 255, 255),
                        -1,
                    )
                    cv2.putText(
                        image,
                        "Right Front",
                        (int(right_front[0]), int(right_front[1]) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    # Display the image
                    cv2.imshow("Image with Left and Right Front Corners", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                return result
            else:
                print("Image doesn't exist")
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
