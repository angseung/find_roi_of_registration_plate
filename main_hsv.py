import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# 색상 범위 설정
lower_orange = (100, 200, 200)
upper_orange = (140, 255, 255)

# lower_green = (30, 80, 80)
lower_green = (20, 80, 80)
# upper_green = (70, 255, 255)
upper_green = (90, 255, 255)

lower_blue = (0, 180, 55)
upper_blue = (20, 255, 200)

# 이미지 파일을 읽어온다
# img = mpimg.imread("img_green/2016_07_13_18345_1528180829T18345S06M2018Y154029E564.jpg", cv2.IMREAD_COLOR)
img = mpimg.imread(
    "img_green/2018_04_22_75122_1527577952T75122S05M2018Y161232E223.jpg",
    cv2.IMREAD_COLOR,
)
# img = mpimg.imread("img_green/2018_04_22_75122_1527577958T75122S05M2018Y161238E850.jpg", cv2.IMREAD_COLOR)
# img = mpimg.imread("img_green/2017_01_05_32951_1527844136T32951S06M2018Y180856E569_1.jpg", cv2.IMREAD_COLOR)

# BGR to HSV 변환
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 색상 범위를 제한하여 mask 생성
img_mask = cv2.inRange(img_hsv, lower_green, upper_green)

# 원본 이미지를 가지고 Object 추출 이미지로 생성
img_result = cv2.bitwise_and(img, img, mask=img_mask)

# 결과 이미지 생성
imgplot = plt.imshow(img_result)

plt.show()
