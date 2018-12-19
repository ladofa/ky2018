#OpenCV 실습
import cv2

print('실습 1')
image = cv2.imread('test.jpg')

cv2.imshow('title', image)
cv2.waitKey(0)
cv2.imwrite('test2.jpg', image)
print(image.shape)

print('실습 2')
blue = image[:, :, 0]
green = image[:, :, 1]
red = image[:, :, 2]

cv2.imshow('blue', blue)
cv2.imshow('green', green)
cv2.imshow('red', red)
cv2.waitKey(0)

image_rev = image[:, :, [2, 1, 0]]
cv2.imshow('rev', image_rev)
cv2.waitKey(0)

print('실습 3')
small = cv2.resize(image, (100, 100))
cv2.imshow('small', small)
cv2.waitKey(0)

print('문제 1')
rect = {'x' : 10, 'y' : 10, 'w' : 100, 'h' : 100}
print('사각형', rect, '가 주어졌을 때')
print('사각형 영역의 이미지만 잘라내어 화면에 표시하시오.')
print('')
print('문제 2')
print('이미지에서', rect, '영역을 까만색으로 채우시오.')
print('(OpenCV 라이브러리가 아닌 numpy를 사용)')
print('')
print('문제 3')
print('lee.jpg파일을 불러온 뒤')
print('원래 이미지의 가운데에 붙이시오')
print('')
print('문제 4')
print('image의 전체 영상을 80% 어둡게 만드시오')