import cv2
import numpy as np
import time

# cap = cv2.VideoCapture('3lb_30secs.mp4')
cap = cv2.VideoCapture(1)

out = cv2.VideoWriter(
    'fgbg_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (1000, 1000))

count = 0
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=150, detectShadows=False)
t0 = time.time()
while cap.isOpened() and count < 400:

    count += 1
    if count % 100 == 0:
        print(count)
    ret, frame = cap.read()
    if not ret:
        break

    # t0 = time.time()
    fgmask = fgbg.apply(frame)
    # cv2_imshow(fgmask)
    # out.write(cv2.resize(fgmask, (1000, 1000)))
    # the output video isn't viewable, apply the fgmask to the original frame
    out.write(cv2.resize(cv2.bitwise_and(
        frame, frame, mask=fgmask), (1000, 1000)))
    # print(time.time()-t0)

print(time.time()-t0)
print("Ms per frame: %s" % ((time.time()-t0)*1000/count))

cap.release()
cv2.destroyAllWindows()
out.release()
