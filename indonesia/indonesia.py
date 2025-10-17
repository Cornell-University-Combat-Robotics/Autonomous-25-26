import cv2
import os
import numpy as np
from typing import Optional, Tuple
import glob

DISPLAY = False


class Indonesia:
    """
    Detect orientation of a robot using color-based centroid detection
    of top (red) and bottom (white) halves of the robot.
    """


    def __init__(self, selected_colors: list[Tuple[int, int, int]], display_final_image: bool = False):
        self.selected_colors = selected_colors
        self.display_final_image = display_final_image
        self.bots = []

    def set_bots(self, bots: dict):
        self.bots = bots

    def get_contours_per_color(self, hsv_image: np.ndarray, color_index: int):
        print ("am i here get countours per color")
        selected_color = self.selected_colors[color_index]
        h0, s0, v0 = selected_color
        print (selected_color)

        hue_radius = 5          # was 10
        s_min = 120             # was 50
        v_min = 120             # was 50

        lower_limit = np.array([max(0, h0 - hue_radius), s_min, v_min], dtype=np.uint8) #TODO: minus nonetype er
        upper_limit = np.array([min(179, h0 + hue_radius), 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, mask

    def find_centroid(self, hsv_image: np.ndarray, color_index: int) -> Tuple[Optional[np.ndarray], np.ndarray]:
        contours, mask = self.get_contours_per_color(hsv_image, color_index)
        if not contours:
            return None, mask

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 10:
            return None, mask

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, mask

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return np.array([cx, cy], dtype=int), mask

    def compute_angle_from_vector(self, dx: float, dy: float) -> float:
        """Angle in degrees measured from +x axis (rightwards), flipped for image y-axis."""

        if dx is None or dy is None:
            print("Warning: compute_angle_from_vector received None input(s).")
            return None  # prevents np.arctan2 crash
    
        dy = -dy  # flip y for image coordinates
        angle_rad = np.arctan2(dy, dx)
        return float(np.degrees(angle_rad) % 360)

    def detect_our_robot_main(self, bot_images: list[np.ndarray]):
        """Chooses the image with the largest combined area of top (color 0) and bottom (color 1)."""
        if not bot_images or not all(img is not None for img in bot_images):
            print("No valid bot images found.")
            return None

        best_img = None
        largest_total_area = 0
        top_back_pixels = [0, 0]

        for img in bot_images:
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            total_area = 0
            # Combine the contour areas for color indices 0 and 1
            for color_index in [0, 1]:
                try:
                    contours, _ = self.get_contours_per_color(hsv_image, color_index)
                except IndexError:
                    print(f"Warning: selected_colors missing index {color_index}")
                    continue
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    total_area += area

                top_back_pixels[color_index] = total_area

            if total_area > largest_total_area and top_back_pixels[0] > 50 and top_back_pixels[1] > 50:
                largest_total_area = total_area
                best_img = img

        if best_img is None:
            print("Our robot was not detected in any image.")
            return None

        if self.display_final_image:
            cv2.imshow("our bot", best_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return best_img

    def detect_arrow_angle(self, image: np.ndarray) -> Optional[float]:
        """
        Detect orientation of the robot as the angle of vector
        from bottom-half (white) to top-half (red).
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        top_center, mask_top = self.find_centroid(hsv_image, 0)    # red
        bottom_center, mask_bottom = self.find_centroid(hsv_image, 1)  # white

        if self.display_final_image:
            # DISPLAY MASKS EACH FRAME
            cv2.imshow("mask_top", mask_top)
            cv2.waitKey(0)
            cv2.imshow("mask_bottom", mask_bottom)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("top_center: " + str(top_center))
        print("bottom_center: " + str(bottom_center))

        if top_center is None or bottom_center is None:
            print("Failed to find both top and bottom centroids.")
            return None

        dx = top_center[0] - bottom_center[0]
        dy = top_center[1] - bottom_center[1]
        angle = self.compute_angle_from_vector(dx, dy)

        if angle is None:
            print("Warning: angle computation failed due to invalid dx/dy.")
            return None
        print("angle: " + str(angle))
        return float(angle)

    def indonesia_main(self, bot_images):
        print ("hi")
        if isinstance(self.bots, dict):
            print ("hi 2")
            bot_images = [bot["img"] for bot in self.bots.get("bots", [])]
        elif isinstance(self.bots, list):
            print ("hi 3")
            bot_images = [bot["img"] for bot in self.bots if "img" in bot]
        else:
            print("Invalid bots format.")
            return {}
        
        print ("hello")
        image = self.detect_our_robot_main(bot_images)
        print ("hello 2")
        if image is None:
            return {}
        
        print ("well hello there")
        orientation = self.detect_arrow_angle(image)
        print("1")
        huey_bbox = None
        for bot_data in self.bots["bots"]:
            if bot_data["img"] is image:
                huey_bbox = bot_data["bbox"]
                print("2")
                break

        print("3")
        huey = {
            "bbox": huey_bbox,
            "orientation": orientation,
            "center": np.mean(huey_bbox, axis=0) if huey_bbox is not None else None,
        }
        print("4")

        enemy_bots = {}
        if isinstance(self.bots, dict) and "bots" in self.bots:
            print("5")
            for bot_data in self.bots["bots"]:
                if bot_data["img"] is not image:
                    enemy_bots = {
                        "bbox": bot_data["bbox"],
                        "center": np.mean(bot_data["bbox"], axis=0),
                    }
                    print("6")
                    break

        result = {"huey": huey, "enemy": enemy_bots}
        print("result: " + str(result))
        return result


if __name__ == "__main__":
    bot_image_folder = os.path.join(os.getcwd(), "test2")
    selected_colors_file = os.path.join(os.getcwd(), "selected_colors.txt")

    bot_image_paths = glob.glob(os.path.join(bot_image_folder, "*.png"))
    bot_images = [cv2.imread(path) for path in bot_image_paths if cv2.imread(path) is not None]

    if not bot_images:
        raise ValueError(f"No valid images found in folder: {bot_image_folder}")

    selected_colors: list[Tuple[int, int, int]] = []
    if not os.path.exists(selected_colors_file):
        raise FileNotFoundError(f"Missing selected colors file: {selected_colors_file}")

    with open(selected_colors_file, "r") as file:
        for line in file:
            hsv = list(map(int, line.strip().split(", ")))
            selected_colors.append((hsv[0], hsv[1], hsv[2]))

    idn = Indonesia(selected_colors, display_final_image=DISPLAY)
    bots = {
        "bots": [
            {
                "img": img,
                "bbox": np.array([
                    [10 + i*10, 10 + i*5],
                    [20 + i*10, 10 + i*5],
                    [20 + i*10, 20 + i*5],
                    [10 + i*10, 20 + i*5]
                ])
            }
            for i, img in enumerate(bot_images)
        ]
    }
    idn.set_bots(bots)
    result = idn.indonesia_main(bot_images)
    print("result:", result)
