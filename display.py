import cv2
import numpy as np


def sin2d(x, y):
    return np.sin(x) + np.cos(y)


w = 512
h = 512
xx, yy = np.meshgrid(np.linspace(0, 0.2 * np.pi, w), np.linspace(0, 2 * np.pi, h))
z = sin2d(xx, yy)
print(z.shape)
z = cv2.normalize(z, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
print(z.shape)


while True:
    z = np.roll(z, (1, 2), (0, 1))
    cv2.imshow("image", z)
    cv2.waitKey(1)
