import math
import numpy as np
import cv2
import time
import random

"""
Image is an np.array of pixels in BGR, Gaussian blur takes in RGB, so that may have to change. 
As for now though, it seems to not mess with output. Try borderType=cv2.BORDER_REPLICATE at some point
"""
def unsharp_mask(image, DISPLAY):
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    
    KERNEL_SIZE = (0, 0)
    SIGMA_X = 5
    start_time = time.time()
    
    # Blur the image using a Gaussian filter
    blurred_image = cv2.GaussianBlur(src = image, ksize = KERNEL_SIZE, sigmaX = SIGMA_X) 
    # Create the unsharp mask by subtracting the blurred image from the original
    unsharp_mask = cv2.addWeighted(src1 = image, alpha = 3.0, src2 = blurred_image, beta = -2.0, gamma = 0)
    end_time = time.time()
    print("Unsharp Mask Time: " + str(((end_time - start_time) * 1000)))
    if DISPLAY:
        vis = np.concatenate((image, unsharp_mask), axis=1)
        cv2.imshow("Side by Side", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return unsharp_mask

if __name__ == "__main__":
    image = cv2.imread("test_files/test6_7?.png")
    length = unsharp_mask(image, True)

    print(f'Unsharp Mask Timing: {length}')
