import glob
import os
import cv2

l = glob.glob('training_bag/**/*.png')
for f in l:
    img = cv2.imread(f)
    cv2.imwrite(f[:-3] + 'jpg', img)
    os.remove(f)
