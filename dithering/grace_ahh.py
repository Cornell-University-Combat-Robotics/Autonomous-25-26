import cv2
import numpy as np
import time

image = cv2.imread("test_files/test3.png")

""" 1 """
start_time = time.time()
# Gaussian blur
blur_1 = cv2.GaussianBlur(image, (0, 0), sigmaX=5)
# Unsharp mask formula
sharpened_1 = cv2.addWeighted(image, 3.0, blur_1, -2.0, 0)
end_time = time.time()
print("Result 1: " + str((end_time - start_time) * 1000))

# """ 2 """
# start_time = time.time()
# # Gaussian blur
# blur_2 = cv2.GaussianBlur(image, (0, 0), sigmaX=5) # 21
# # Unsharp mask formula
# # TODO: WHY DO WE SKIP A LAYER HERE AND IT WORKS BETTER AHHHHHHHH
# sharpened_2 = cv2.addWeighted(image, 2, blur_2, -1, 0)
# end_time = time.time()
# print("Result 2: " + str((end_time - start_time) * 1000))

# """ 3 """
# start_time = time.time()
# # Gaussian blur
# blur_3 = cv2.GaussianBlur(image, (0, 0), sigmaX=5) # 21
# # Unsharp mask formula
# # TODO: WHY DO WE SKIP A LAYER HERE AND IT WORKS BETTER AHHHHHHHH
# sharpened_3 = cv2.addWeighted(image, 2.7, blur_3, -1.8, 0)
# end_time = time.time()
# print("Result 3: " + str((end_time - start_time) * 1000))

# # cv2.imshow("Sharpened", sharpened)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# print("1st and 2nd equal: " + str(np.array_equal(sharpened_1,sharpened_2)))
# print("1st and 3rd equal: " + str(np.array_equal(sharpened_1,sharpened_3)))
# print("2nd and 3rd equal: " + str(np.array_equal(sharpened_2,sharpened_3)))

# vis = np.concatenate((image, sharpened_1, sharpened_2, sharpened_3), axis=1)
# cv2.imshow("Side by Side", vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

vis = np.concatenate((image, sharpened_1), axis=1)
cv2.imshow("Side by Side", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()




