import cv2
import numpy as np
from corner_detection.color_picker import ColorPicker
import time

"""
    Takes in BGR image, converts to Lab, then snaps colors to color picker, 
    and makes other pixels either unchanged or black. Converts back to BGR, 
    and returns.

    Parameters
    ----------
    img_bgr : np.ndarray
        HxWx3 uint8 OpenCV image in BGR.
    robot_colors_bgr : np.ndarray
        Nx3 uint8 array of BGR robot colors (e.g. N = 3 or 4).
    thresh_lab : float
        Distance threshold in Lab space for snapping to robot colors.
    keep_background : bool
        If True: pixels not close to any robot color stay as in the original image.
        If False: pixels not close to any robot color become black.

    Returns
    -------
    out_img : np.ndarray
        Quantized image in BGR.
    palette_bgr : np.ndarray
        The robot colors used (same as robot_colors_bgr, cast to uint8).
    """
def quantize_robot_colors(
    img_bgr,
    robot_colors_bgr,
    thresh_lab=25.0,
    keep_background=True,
    show = False,
    custom_weights = None,
):
    H, W = img_bgr.shape[:2]
    N = H * W
    
    #OG Goat params from Ryan Tuning, best for Huey v Prince
    # thresh_lab = 24.725 #Ryan god tuning
    # weights = np.array([0.171, 0.3777, 0.669], dtype=np.float32) # Ryan god tuning
    
    # Define weights for L, a, and b
    # Setting L_weight to 0.0 ignores brightness entirely.
    # Setting it to 0.2 makes it matter, but much less than color.
    L_weight = 0.05
    weights = np.array([L_weight, 1.0, 1.0], dtype=np.float32)  # WEIGHTS: CIELAB color space
    if custom_weights:
        L_weight = custom_weights[0]
        weights = np.array(custom_weights, dtype=np.float32)
    
    # Convert image to Lab
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    flat_lab = img_lab.reshape(-1, 3).astype(np.float32)  # (N, 3)

    # Convert robot colors to Lab
    robot_bgr_1x = robot_colors_bgr.reshape(1, -1, 3)
    robot_lab_1x = cv2.cvtColor(robot_bgr_1x, cv2.COLOR_BGR2LAB)
    robot_lab = robot_lab_1x.reshape(-1, 3).astype(np.float32)  # (M, 3), M=#colors

    # Distance to robot colors (no big broadcast, no sqrt)
    thresh2 = thresh_lab * thresh_lab
    flat = flat_lab  # (N, 3)

    # Start with first color (weighted)
    diff0 = (flat - robot_lab[0]) * weights
    min_dist2 = np.sum(diff0 ** 2, axis=1)
    min_idx = np.zeros_like(min_dist2, dtype=np.int32)

    # Compare with remaining robot colors (weighted)
    for i in range(1, robot_lab.shape[0]):
        diff = (flat - robot_lab[i]) * weights
        d = np.sum(diff ** 2, axis=1)
        
        mask = d < min_dist2
        min_dist2[mask] = d[mask]
        min_idx[mask] = i

    # # Start with first color
    # min_dist2 = np.sum((flat - robot_lab[0]) ** 2, axis=1)
    # min_idx = np.zeros_like(min_dist2, dtype=np.int32)

    # # Compare with remaining robot colors
    # for i in range(1, robot_lab.shape[0]):
    #     d = np.sum((flat - robot_lab[i]) ** 2, axis=1)
    #     mask = d < min_dist2
    #     min_dist2[mask] = d[mask]
    #     min_idx[mask] = i

    # Pixels close enough to some robot color
    mask_robot = min_dist2 < thresh2  # shape (N,)

    # Build output image
    if keep_background:
        # Start from original image
        out_img = img_bgr.copy()
    else:
        # Everything black by default
        out_img = np.zeros_like(img_bgr)

    flat_out = out_img.reshape(-1, 3)

    # Snap robot pixels to their nearest robot color (in BGR)
    robot_colors_bgr = robot_colors_bgr.astype(np.uint8)
    flat_out[mask_robot] = robot_colors_bgr[min_idx[mask_robot]]
    if show:
        out_img = cv2.resize(out_img,None, fx=3.0,fy=3.0,interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Quantized Image", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out_img



if __name__ == "__main__":
    dummy = np.zeros((8, 8, 3), dtype=np.uint8)
    _ = cv2.cvtColor(dummy, cv2.COLOR_BGR2LAB)
    _ = cv2.cvtColor(dummy, cv2.COLOR_BGR2HSV)
    
    img = cv2.imread("quantization/test_files/test3.png")
    img = cv2.resize(img,None, fx = 3.0, fy=3.0,interpolation=cv2.INTER_CUBIC)
    # Pick colors (HSV) using your ColorPicker
    color_picker = ColorPicker
    colors_hsv = np.array(color_picker.pick_colors(img), dtype=np.uint8)

    if colors_hsv.size == 0:
        print("No colors selected, exiting.")
        exit(0)

    # Convert picked HSV colors to BGR
    colors_hsv_1x = colors_hsv.reshape(1, -1, 3)
    bgr_colors_1x = cv2.cvtColor(colors_hsv_1x, cv2.COLOR_HSV2BGR)
    bgr_colors = bgr_colors_1x.reshape(-1, 3)  # (N_colors, 3)

    # outer timing for whole call
    start_time = time.time()
    out_img = quantize_robot_colors(
        img,
        bgr_colors,
        thresh_lab=35,
        keep_background=True,
        show=True
    )
    total_time_ms = (time.time() - start_time) * 1000.0
    print(f"Total quantize_robot_colors call (outer): {total_time_ms:.3f} ms")

    cv2.imshow("Original Image", img)
    cv2.imshow("Quantized Image", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
