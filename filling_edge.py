import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read image as cv2.COLOR_BGR
# img = mpimg.imread("img_green/2018_04_22_75122_1527577952T75122S05M2018Y161232E223.jpg", cv2.IMREAD_COLOR)
img = mpimg.imread(
    "img_green/2017_01_05_32951_1527844136T32951S06M2018Y180856E569_1.jpg",
    cv2.IMREAD_COLOR,
)
img = cv2.resize(img, (400, 300))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)

threshold1 = 200
threshold2 = 300
edge_img = cv2.Canny(blurred, threshold1, threshold2)

# _, thresh = cv2.threshold(edge_img, 128, 255, cv2.THRESH_BINARY)

# get the (largest) contour
contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for d in contours:
    (x, y, w, h) = cv2.boundingRect(d)
    cv2.rectangle(
        img,
        pt1=(x, y),
        pt2=(x + w, y + h),
        color=(255, 255, 255),
        thickness=2,
    )
# contours = contours[0] if len(contours) == 2 else contours[1]
# big_contour = max(contours, key=cv2.contourArea)
# cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
# print(len(contours), "objects were found in this image.")

# draw white filled contour on black background
# result = np.zeros_like(img)
# cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)

# save results
# cv2.imwrite('knife_edge_result.jpg', result)

# 결과 이미지 생성
imgplot = plt.imshow(img)

plt.show()
