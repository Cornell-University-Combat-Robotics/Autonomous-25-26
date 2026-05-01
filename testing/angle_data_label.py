import os
import cv2
import csv
import math
import numpy as np

# Settings
DATA_SET_NAME = "bluey"   # Name of your dataset folder in "testing_data"
MANUAL_CENTER = False         # Allows you to select the center of the bot by hand 

# --- Change this to your folder path ---
ALL_TRAINING_DATA_PATH = "testing_data/"
FOLDER_PATH = os.path.join(ALL_TRAINING_DATA_PATH, DATA_SET_NAME)

# Output CSV path
CSV_OUTPUT = os.path.join(FOLDER_PATH, "angles_output.csv")

# Valid image file extensions
IMAGE_EXTENSIONS = (".png",)

# Global state for mouse callback
state = {
    "angle": None,
    "tail": None,       # First click: tail of the vector
    "base_image": None,
    "window_name": "Image Viewer",
}

def draw_overlay(img, tail=None, angle=None):
    """Draw tail marker and angle line on a copy of the image."""
    overlay = img.copy()

    if tail is not None:
        cx, cy = tail

        # Draw crosshair at tail
        cross_size = 20
        color_cross = (0, 255, 0)
        cv2.line(overlay, (cx - cross_size, cy), (cx + cross_size, cy), color_cross, 2)
        cv2.line(overlay, (cx, cy - cross_size), (cx, cy + cross_size), color_cross, 2)
        cv2.circle(overlay, tail, 5, color_cross, -1)

        if angle is not None:
            # Draw angle line from tail outward toward head
            length = min(img.shape[0], img.shape[1]) // 3
            rad = math.radians(angle)
            ex = int(cx + length * math.cos(rad))
            ey = int(cy - length * math.sin(rad))  # y-axis inverted in image coords
            cv2.line(overlay, tail, (ex, ey), (0, 100, 255), 2)
            cv2.circle(overlay, (ex, ey), 6, (0, 100, 255), -1)

            # Display angle text
            label = f"Angle: {angle:.1f} deg"
            cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 100, 255), 1, cv2.LINE_AA)

    # Instructions
    instructions = [
        "Click 1: set the tail" if MANUAL_CENTER else None,
        "Click 2: set the head" if MANUAL_CENTER else "Click: set angle",
        "ENTER: confirm & next",
        "S: skip image",
        "Q: quit"
    ]

    for i, text in enumerate(instructions):
        y = overlay.shape[0] - 15 - i * 22
        cv2.putText(overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)

    return overlay

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if MANUAL_CENTER and state["tail"] is None:
                # First click: set the tail point (< head ---------- tail )
                state["tail"] = (x, y)
                updated = draw_overlay(state["base_image"], state["tail"], None)
                cv2.imshow(state["window_name"], updated)
        else:
            # Second click: set the head point and compute angle
            if MANUAL_CENTER:
                tx, ty = state["tail"]
            else:
                tx, ty = state["center"] 

            dx = x - tx
            dy = ty - y  # flip y for standard angle convention (up = positive)
            angle = math.degrees(math.atan2(dy, dx)) % 360
            state["angle"] = angle

            if MANUAL_CENTER:
                updated = draw_overlay(state["base_image"], state["tail"], state["angle"])
            else:
                updated = draw_overlay(state["base_image"], state["center"], state["angle"])

            cv2.imshow(state["window_name"], updated)


def main():
    results = []

    image_files = sorted([
        f for f in os.listdir(FOLDER_PATH)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    if not image_files:
        print("No images found in folder.")
        return

    cv2.namedWindow(state["window_name"])
    cv2.setMouseCallback(state["window_name"], mouse_callback)

    for filename in image_files:
        image_path = os.path.join(FOLDER_PATH, filename)
        print(f"\nProcessing: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load: {image_path}")
            continue

        # Upscale to fit target window size
        TARGET_WIDTH = 600
        TARGET_HEIGHT = 600
        h, w = image.shape[:2]
        scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
        if scale > 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # This is the center of the screen
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        state["center"] = center

        # Reset state for this image
        state["angle"] = None
        state["tail"] = None
        state["base_image"] = image.copy()

        # Show initial overlay (no angle yet)
        display = draw_overlay(image, None, None)
        cv2.imshow(state["window_name"], display)

        print(f"Showing: {filename}  |  Click to select angle, ENTER to confirm, S to skip, Q to quit.")

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == 13 or key == 10:  # ENTER key
                if state["angle"] is not None:
                    print(f"  -> Saved angle: {state['angle']:.1f} deg")
                    results.append({"filename": filename, "angle": round(state["angle"], 2)})
                else:
                    print("  -> No angle selected; skipping.")
                break

            elif key == ord('s'):  # Skip
                print("  -> Skipped.")
                break

            elif key == ord('q'):  # Quit
                print("Quitting early.")
                cv2.destroyAllWindows()
                _write_csv(results)
                return

    cv2.destroyAllWindows()
    _write_csv(results)


def _write_csv(results):
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "angle"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} entries to: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()