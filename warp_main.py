import os
import cv2
import numpy as np
import torch

ARENA_WIDTH = 700

"""
Duplicated from vid_and_img_processing/vid_to_warped_frames.py

Given some frame of the arena, allows user to select points. 
Returns the resulting homography matrix.
Params:
    - frame: A cv2 image of the full arena, as seen from the camera
    - output_w: The ideal output width of the image. By default, 700px
    - output_h: The ideal output height of the image. By default, 700px

Returns:
    - A numpy matrix, which, when used with cv2.warpPerspective
    - 'flattens' the perspective

As a change from the original get_homography_mat, does NOT resize the input image.
"""
folder = os.getcwd() + "/main_files"

import numpy as np
import cv2
import os

def get_homography_mat(frame, display_scale=1.0):
    corners = []
    outer_padding = 50
    clickable_padding = 150
    total_padding = clickable_padding + outer_padding

    # Add black border around the frame.
    padded_frame = cv2.copyMakeBorder(frame, total_padding, total_padding, total_padding, total_padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Resize for display
    h, w = padded_frame.shape[:2]
    display_frame = cv2.resize(padded_frame, (int(w * display_scale), int(h * display_scale)))

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map display coordinates back to original frame coordinates
            orig_x = int(x / display_scale)
            orig_y = int(y / display_scale)

            if orig_x < outer_padding or orig_y < outer_padding or orig_x >= padded_frame.shape[1] - outer_padding or orig_y >= padded_frame.shape[0] - outer_padding:       
                print(f"Clicked outside valid area: ({orig_x}, {orig_y})")
                return

            corners.append([orig_x - outer_padding, orig_y - outer_padding])  # Save coords relative to original frame
            print(f"Point added: {orig_x - outer_padding}, {orig_y - outer_padding}")
            draw_corners()

    def draw_corners():
        frame_copy = display_frame.copy()
        for point in corners:
            # Map original coordinates to display coordinates for drawing
            draw_x = int((point[0] + outer_padding) * display_scale)
            draw_y = int((point[1] + outer_padding) * display_scale)
            draw_point = (draw_x, draw_y)
            cv2.circle(frame_copy, draw_point, 5, (0, 255, 0), -1)
            cv2.putText(frame_copy, str(point), draw_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("Warp: Select arena corners from top left, top right, bottom right to bottom left. Press 'z' to undo click", frame_copy)

    cv2.imshow("Warp: Select arena corners from top left, top right, bottom right to bottom left. Press 'z' to undo click", display_frame)
    cv2.setMouseCallback("Warp: Select arena corners from top left, top right, bottom right to bottom left. Press 'z' to undo click", click_event)

    key = cv2.waitKey(1) & 0xFF
    while len(corners) < 4 and key != 27:
        if key == ord('z'):
            if corners:
                removed = corners.pop()
                print(f"Point removed: {removed}")
                draw_corners()
            else:
                print("No points to remove.")
        key = cv2.waitKey(1) & 0xFF

    print("Final Selected Points:", corners)
    dest_pts = [[0, 0], [ARENA_WIDTH, 0], [ARENA_WIDTH, ARENA_WIDTH], [0, ARENA_WIDTH]]
    matrix, _ = cv2.findHomography(np.array(corners), np.array(dest_pts))
    cv2.destroyAllWindows()

    # 1. Create a translation matrix to account for the "missing" padding
    # This shifts the coordinate system of the homography
    T = np.array([[1, 0, clickable_padding],
                  [0, 1, clickable_padding],
                  [0, 0, 1]], dtype=np.float32)
    
    # 2. Combine them: New_H = H * T
    # This effectively tells the warp to look "outside" the 0,0 bounds
    matrix = matrix @ T


    output_file = folder + "/homography_matrix.txt"
    with open(output_file, "w") as file:
        for row in matrix:
            file.write(", ".join(map(str, row)) + "\n")
    print(f"Homography matrix has been saved to '{output_file}'.")

    return matrix

"""
Re-combined from camera_test/warp.py, vid_and_img_processing/warp_image.py

Given a frame and a homography matrix, warp the perspective and isolate the flat plane.

Params:
    - frame: A cv2 image of the full arena, as seen from the camera
    - h_mat: The homography matrix used for transformation, derived from 'get_homography_mat'
    - output_w: The ideal output width of the image. By default, 700px
    - output_h: The ideal output height of the image. By default, 700px

Returns:
    - A cv2 image of the 'warped' arena

As a change from the original warp, does NOT resize the input image.
"""
# def warp(frame, h_mat):
#     clickable_padding = 150 
#     padded_frame = cv2.copyMakeBorder(frame, clickable_padding, clickable_padding, clickable_padding, clickable_padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
#     if torch.cuda.is_available():
#         gpu_frame = cv2.UMat(padded_frame)
#         return cv2.warpPerspective(gpu_frame, h_mat, (ARENA_WIDTH, ARENA_WIDTH)).get()
#     else:
#         frame = padded_frame
#         return cv2.warpPerspective(frame, h_mat, (ARENA_WIDTH, ARENA_WIDTH))

def warp(frame, h_mat):

    # DEPRECATED: Use warp_map with precomputed maps instead for better performance if doing multiple warps with the same homography
    print("WARNING: Using warp() which is significantly slower than warp_map() with precomputed maps. Consider using get_warp_maps() and warp_map() for better performance if warping multiple frames with the same homography.")

    if torch.cuda.is_available():
        gpu_frame = cv2.UMat(frame)
        return cv2.warpPerspective(
            gpu_frame, 
            h_mat, 
            (ARENA_WIDTH, ARENA_WIDTH),
            flags=cv2.INTER_NEAREST,        # Keeps your clusters sharp, try INTER_CUBIC
            borderMode=cv2.BORDER_CONSTANT,   # Fills the "off-camera" corners with black
            borderValue=(0, 0, 0)
        ).get()
    # 3. Perform the warp in one shot
    else:
        return cv2.warpPerspective(
            frame, 
            h_mat, 
            (ARENA_WIDTH, ARENA_WIDTH),
            flags=cv2.INTER_NEAREST,        # Keeps your clusters sharp, try INTER_CUBIC
            borderMode=cv2.BORDER_CONSTANT,   # Fills the "off-camera" corners with black
            borderValue=(0, 0, 0)
        )

def get_warp_maps(h_mat, dst_size=(ARENA_WIDTH, ARENA_WIDTH)):
    # Calculate the inverse homography
    h_inv = np.linalg.inv(h_mat)
    
    # Create a grid of coordinates for the destination image
    w, h = dst_size
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Flatten and add the '1' for homogeneous coordinates
    coords = np.stack([grid_x.ravel(), grid_y.ravel(), np.ones_like(grid_x.ravel())])
    
    # Map back to the source image coordinates
    src_coords = h_inv @ coords
    src_coords /= src_coords[2] # Normalize (Perspective division)
    
    # Reshape back to the image grid
    map_x = src_coords[0].reshape(h, w).astype(np.float32)
    map_y = src_coords[1].reshape(h, w).astype(np.float32)
    
    return map_x, map_y

def warp_map(frame, map_x, map_y):
    # This is significantly faster than warpPerspective for repeated transforms
    return cv2.remap(
        frame, 
        map_x, 
        map_y, 
        interpolation=cv2.INTER_NEAREST, # Keeps clusters sharp
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0)
    )


if __name__ == "__main__":
    frame = cv2.imread('./vid_and_img_processing/sample_cage_ss.png')
    h_mat = get_homography_mat(frame)
    warped_frame = warp(frame, h_mat)
    cv2.imshow("Warped cage", warped_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    warped_frame = warp(frame, h_mat)
    cv2.imshow("Warped cage", warped_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
