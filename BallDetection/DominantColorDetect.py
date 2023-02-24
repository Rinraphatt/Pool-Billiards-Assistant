import cv2 as cv
import numpy as np

img = cv.imread('../pics/pool_table_noBall.jpg')

height, width, _ = np.shape(img)

def createBar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)

data = np.reshape(img, (height * width, 3))
data = np.float32(data)

number_clusters = 3
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv.kmeans(data, number_clusters, None, criteria, 10, flags)
print(centers)

bars = []
rgbValues = []

for index, row in enumerate(centers):
    bar, rgb = createBar(200, 200, row)
    bars.append(bar)
    rgbValues.append(rgb)

imgBar = np.hstack(bars)

for index, row in enumerate(rgbValues):
    image = cv.putText(imgBar, f'{index + 1}. RGB: {row}', (5 + 200 * index, 200 - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
    print(f'{index + 1}. RGB: {row}')

print(f'RGB1TO2 : {abs(rgbValues[0][0]-rgbValues[1][0])+abs(rgbValues[0][1]-rgbValues[1][1])+abs(rgbValues[0][2]-rgbValues[1][2])}')
print(f'RGB1TO3 : {abs(rgbValues[0][0]-rgbValues[2][0])+abs(rgbValues[0][1]-rgbValues[2][1])+abs(rgbValues[0][2]-rgbValues[2][2])}')

cv.imshow("Image", img)
cv.imshow("rgbBar", imgBar)

cv.waitKey(0)

