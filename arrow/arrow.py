import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple
import glob

DISPLAY = True


class Arrow:
    """
    Detect orientation of a robot-like arrow using color-based centroid detection
    and PCA on the arrow mask.
    """

    def __init__(self, selected_colors: list[Tuple[int, int, int]], display_final_image: bool = False):
        self.selected_colors = selected_colors
        self.display_final_image = display_final_image
        self.bots = []

    def set_bots(self, bots: list[dict]):
        self.bots = bots

    def detect_our_robot_main(self, bot_images: list[np.ndarray]):
        """
        Detects the image containing our robot using the arrow color.
        Chooses the image with the largest detected contour.
        """
        try:
            if not bot_images or not all(img is not None for img in bot_images):
                print("No valid bot images found.")
                return None

            arrow_color_index = 1  # arrow color in selected_colors
            best_img = None
            largest_area = 0

            for img in bot_images:
                # cv2.imshow("OPTION", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                contours, mask = self.get_contours_per_color(hsv_image, arrow_color_index)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > largest_area:
                        largest_area = area
                        best_img = img

            if best_img is None:
                print("Our robot was not detected in any image.")
            return best_img

        except Exception as e:
            print(f"Unexpected error in detect_our_robot_main: {e}")
            return None

    def get_image_center(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return np.array([w // 2, h // 2], dtype=int)

    def get_contours_per_color(self, hsv_image: np.ndarray, color_index: int):
        selected_color = self.selected_colors[color_index]
        # Hue range +/-10, avoid underflow/overflow
        lower_limit = np.array([max(0, selected_color[0] - 10), 20, 20], dtype=np.uint8)
        upper_limit = np.array([min(179, selected_color[0] + 10), 255, 255], dtype=np.uint8)

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
        # return as [x, y]
        return np.array([cx, cy], dtype=int), mask

    def compute_angle_from_vector(self, dx: float, dy: float) -> float:
        """
        Angle in degrees measured from +x axis, converting image coords
        (y growing downward).
        """
        dy = -dy  # flip y for image coordinates
        angle_rad = np.arctan2(dy, dx)
        return float(np.degrees(angle_rad) % 360)

    def center_of_mass(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # color index 1 is the white arrow center per your earlier code
        center_of_arrow, mask = self.find_centroid(hsv_image, 1)
        return center_of_arrow, mask

    def pca_on_mask(self, bot_image: np.ndarray, mask: np.ndarray, center_of_arrow: np.ndarray):
        """
        Perform PCA on the largest connected component of mask and return
        endpoints for visualization, the principal vector, and an approximate angle
        from the vector from green bbox center to white centroid (if bbox exists).
        """
        # If no arrow centroid or mask, bail out
        if center_of_arrow is None:
            return None

        # get largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            # no components besides background
            return None

        # pick largest non-background label
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        arrow_mask = np.zeros_like(mask)
        arrow_mask[labels == largest_label] = 255

        # if DISPLAY:
        #     cv2.imshow("arrow_mask", arrow_mask)
        #     cv2.waitKey(1)

        # coords are (row, col) -> (y, x)
        coords = np.column_stack(np.where(arrow_mask == 255))
        if coords.shape[0] < 2:
            return None

        # PCA on coords
        pca = PCA(n_components=2)
        pca.fit(coords)
        # component is in [row, col] (y, x) order
        pca_vec = pca.components_[0].copy()  # e.g. [dy, dx] in row/col space

        # Compute an approximate angle using vector from image center (bbox in your earlier code)
        # If you have a green bbox center, you can replace this with that; here we'll use image center
        center_of_bbox = self.get_image_center(bot_image)

        # Convert center_of_arrow (x,y) and center_of_bbox (x,y) to numpy arrays
        # earlier coords used [x, y]
        actual_vector = center_of_arrow - center_of_bbox  # [x, y]
        approx_pca_angle = self.compute_angle_from_vector(float(actual_vector[0]), float(actual_vector[1]))

        # draw centroids (ensure ints)
        cv2.circle(bot_image, (int(center_of_arrow[0]), int(center_of_arrow[1])), 5, (255, 255, 255), -1)
        cv2.circle(bot_image, (int(center_of_bbox[0]), int(center_of_bbox[1])), 5, (0, 255, 0), -1)

        # Scale PCA vector for visualization; note pca_vec is [row, col] -> [dy, dx] so flip when using compute_angle_from_vector
        length = int(max(bot_image.shape[:2]) * 0.4)
        dx = int(pca_vec[1] * length)
        dy = int(pca_vec[0] * length)

        p1 = (int(center_of_arrow[0] - dx), int(center_of_arrow[1] - dy))
        p2 = (int(center_of_arrow[0] + dx), int(center_of_arrow[1] + dy))

        return p1, p2, pca_vec, approx_pca_angle

    def angle_diff(self, a: float, b: float) -> float:
        """Unsigned wrapped difference in degrees"""
        return abs(((a - b + 180) % 360) - 180)

    def detect_arrow_angle(self, image) -> Optional[float]:
        """
        Full detection pipeline: find centroids, run PCA, resolve 180deg ambiguity,
        and return final angle in degrees or None on failure.
        """
        center_of_arrow, mask = self.center_of_mass(image)
        if center_of_arrow is None or mask is None:
            print("Failed to find arrow centroid or mask.")
            return None

        pca_res = self.pca_on_mask(image, mask, center_of_arrow)
        if pca_res is None:
            print("PCA failed or insufficient data.")
            return None

        p1, p2, pca_vec, approx_pca_angle = pca_res

        # compute two candidate angles from PCA principal vector
        pca_angle_1 = self.compute_angle_from_vector(float(pca_vec[1]), float(pca_vec[0]))
        pca_angle_2 = (pca_angle_1 + 180) % 360
        closeness_1 = self.angle_diff(approx_pca_angle, pca_angle_1)
        closeness_2 = self.angle_diff(approx_pca_angle, pca_angle_2)
        pca_angle = pca_angle_1 if closeness_1 < closeness_2 else pca_angle_2

        # visualize results

        if self.display_final_image or DISPLAY:
            cv2.line(image, p1, p2, (0, 0, 0), 2)
            cv2.circle(image, (int(center_of_arrow[0]), int(center_of_arrow[1])), 5, (255, 255, 255), -1)
            cv2.putText(image, f"{pca_angle:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.imshow("bot_image with PCA axis", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        return float(pca_angle)
    
    def arrow_main(self, bot_images):
        if bot_images is None:
            bot_images = [bot["img"] for bot in self.bots]
        
        image = self.detect_our_robot_main(bot_images) # error

        if image is None:
            return {}
        
        # cv2.imshow("My Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        orientation = self.detect_arrow_angle(image) # error
        
        # Find the identified bot (our robot)
        huey_bbox = None
        for bot_data in self.bots['bots']:
            if bot_data["img"] is image:
                huey_bbox = bot_data["bbox"]
                break

        huey = {
            "bbox": huey_bbox,
            "center": self.get_image_center(image),
            "orientation": orientation,
        }

        # Enemy bots are all except the identified bot
        enemy_bots = []
        for bot_data in self.bots['bots']:
            if bot_data["img"] is not image:
                enemy = {
                    "bbox": bot_data["bbox"],
                    "center": np.mean(bot_data["bbox"], axis=0),
                }
                enemy_bots.append(enemy)

        result = {"huey": huey, "enemy": enemy_bots}
        return result

if __name__ == "__main__":
    bot_image_folder = os.path.join(os.getcwd(), "test")
    selected_colors_file = os.path.join(os.getcwd(), "selected_colors.txt")

    bot_image_paths = glob.glob(os.path.join(bot_image_folder, "*.png"))  # or *.jpg etc.
    bot_images = []
    for path in bot_image_paths:
        img = cv2.imread(path)
        if img is not None:
            bot_images.append(img)
        else:
            print(f"Warning: failed to load image {path}")

    if not bot_images:
        raise ValueError(f"No valid images found in folder: {bot_image_folder}")

    # Load selected colors
    selected_colors: list[Tuple[int, int, int]] = []
    if not os.path.exists(selected_colors_file):
        raise FileNotFoundError(f"Missing selected colors file: {selected_colors_file}")

    with open(selected_colors_file, "r") as file:
        for line in file:
            hsv = list(map(int, line.strip().split(", ")))
            selected_colors.append((hsv[0], hsv[1], hsv[2]))
    
    arrow = Arrow(selected_colors, display_final_image=DISPLAY)
    result = arrow.arrow_main(bot_images)
    print("result: " + str(result))
