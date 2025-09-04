import math
import os
import cv2
import numpy as np


class Arrow:
    """
    A class for detecting the orientation of a robot in a given image
    based on its unique corner colors.
    """
    
    def __init__(self, selected_colors, display_final_image=False):
        """
        Initializes the Arrow class.

        Args:
            selected_colors (list): Manually selected HSV colors for corners.
            display_final_image (bool): Whether to display the final image with
                                        centroids and angle overlay.
        """
        self.selected_colors = selected_colors
        self.display_final_image = display_final_image

    def get_image_center(self, image: np.ndarray) -> tuple[int, int]:
        """
        Finds the geometric center of an image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            (int, int): (x, y) coordinates of the image center.
        """
        h, w = image.shape[:2]  # height, width
        center_x = w // 2
        center_y = h // 2
        return (center_x, center_y)

    def get_contours_per_color(self, hsv_image: np.ndarray, color_index: int):
        """
        Retrieves contours for the given color index.

        Args:
            hsv_image (np.ndarray): Input image in HSV format.
            color_index (int): Index in self.selected_colors for the desired color.

        Returns:
            list: Contours corresponding to the given color.
        """
        selected_color = self.selected_colors[color_index]
        print("selected color: " + str(selected_color))

        # Define HSV range
        lower_limit = np.array([max(0, selected_color[0] - 10), 20, 20])
        upper_limit = np.array([min(179, selected_color[0] + 10), 255, 255])

        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow(f"Mask Color {color_index}", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return contours, mask
    
    def find_centroid(self, image: np.ndarray, hsv_image: np.ndarray, color_index: int):
        """
        Finds the centroid of the largest contour for a given color.

        Args:
            image (np.ndarray): BGR image (for drawing).
            hsv_image (np.ndarray): HSV version of the image.
            color_index (int): Which color to use.

        Returns:
            tuple or None: (x, y) centroid or None if not found.
        """
        contours, mask = self.get_contours_per_color(hsv_image, color_index)
        if not contours:
            return None

        # pick largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 10:
            return None

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy)

    @staticmethod
    def compute_angle_between_points(p1: tuple, p2: tuple):
        """
        Computes the angle of the line between two points.

        Args:
            p1 (tuple): First point (x1, y1).
            p2 (tuple): Second point (x2, y2).

        Returns:
            float: Angle in degrees relative to x-axis.
        """
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = -(y2 - y1)  # invert y for image coords
        angle_rad = np.arctan2(dy, dx)
        return math.degrees(angle_rad) % 360

    def arrow_detection_main(self, image: np.ndarray):
        """
        Detects center of mass for the arrow and computes orientation angle.

        Args:
            image (np.ndarray): Input robot image (BGR).

        Returns:
            dict: containing centroids and angle.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        center_of_arrow = self.find_centroid(image, hsv_image, 1)
        return center_of_arrow


if __name__ == "__main__":
    # We assume that we know which bot is Huey
    bot_image_path = os.getcwd() + "/huey.png"
    selected_colors_file = os.getcwd() + "/selected_colors.txt"

    bot_image = cv2.imread(bot_image_path)
    if bot_image is None:
        raise ValueError(f"Failed to load bot image: {bot_image_path}")

    selected_colors = []
    with open(selected_colors_file, "r") as file:
        for line in file:
            hsv = list(map(int, line.strip().split(", ")))
            selected_colors.append(hsv)

    arrow = Arrow(selected_colors, display_final_image=True)
    center_of_bbox = arrow.get_image_center(bot_image) # [x, y] point
    center_of_arrow = arrow.arrow_detection_main(bot_image) # [x, y] point

    result_point = (int(center_of_arrow[0]), int(center_of_arrow[1]))

    cv2.circle(bot_image, result_point, 6, (0, 255, 0), -1)
    cv2.putText(bot_image, "Arrow", (result_point[0] + 10, result_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw geometric center (blue now)
    center_point = (int(center_of_bbox[0]), int(center_of_bbox[1]))
    cv2.circle(bot_image, center_point, 6, (255, 0, 0), -1)  # blue
    cv2.putText(bot_image, "Center", (center_point[0] + 10, center_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    print("center_of_bbox: " + str(center_of_bbox))
    print("center_of_arrow: " + str(center_of_arrow))
    print("compute_angle_between_points: " + str(arrow.compute_angle_between_points(center_point, result_point)))

    # Draw arrow from center to arrow detection (yellow)
    cv2.arrowedLine(bot_image, center_point, result_point, (0, 255, 255), 2, tipLength=0.1)
    
    cv2.imshow("Bot with Arrow", bot_image)
    cv2.waitKey(0)  # wait until a key is pressed
    cv2.destroyAllWindows()

