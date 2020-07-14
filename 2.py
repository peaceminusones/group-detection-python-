import cv2

a = cv2.imread("D:/paper/IEEE model/0 2019 paper/picture/3shatian6_frame_10_1.png")
b = cv2.resize(a,(1080,720),interpolation=cv2.INTER_AREA)
cv2.imshow("b",b)
cv2.imwrite("3shatian6_frame_10_2.png", b)
cv2.waitKey(0) #等待按键