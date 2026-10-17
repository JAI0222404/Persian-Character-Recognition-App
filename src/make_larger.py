import cv2

img = cv2.imread("letter.png")
big = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
cv2.imwrite("letter_big.png", big)