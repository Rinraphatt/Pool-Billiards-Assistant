import cv2 as cv
import math

path = 'pool_table_noBall.jpg'
img = cv.imread(path)
# img = cv.resize(img, (1000, 968))
img = cv.resize(img, (1920, 1080))

pointsList = []

def mousePoints(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN: 
        size = len(pointsList)
        if size != 0 and size % 3 != 0 :
            cv.line(img, tuple(pointsList[round((size-1)/3)*3]), (x,y), (255,0,0), 2)
        cv.circle(img, (x,y), 5, (0,0,255), cv.FILLED)
        pointsList.append([x,y])


def getGradient(pt1, pt2):
    mod = (pt2[0]-pt1[0])

    if mod == 0:
        mod = 0.01

    return (pt2[1]-pt1[1]) / mod


def getAngle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:]
    m1 = getGradient(pt1, pt2)
    m2 = getGradient(pt1, pt3)
    angR = math.atan((m1+m2)/(1+m1*m2))
    angD = abs(round(math.degrees(angR)))
    # print(angD)

    print(pointsList[-3:])
    if ((pt2[1] <= pt1[1]) and (pt3[1] >= pt1[1])) or ((pt3[1] <= pt1[1]) and (pt2[1] >= pt1[1])):
        cv.circle(img, (pt1[0],pt1[1]), 5, (0,255,0), cv.FILLED)
        cv.circle(img, (2*pt1[0] - pt2[0],pt2[1]), 5, (0,255,0), cv.FILLED)
        cv.circle(img, (2*pt1[0] - pt3[0],pt3[1]), 5, (0,255,0), cv.FILLED)
        cv.line(img, (pt1[0], pt1[1]), (2*pt1[0] - pt2[0], pt2[1]), (0,255,0), 2)
        cv.line(img, (pt1[0], pt1[1]), (2*pt1[0] - pt3[0], pt3[1]), (0,255,0), 2)
    else:
        cv.circle(img, (pt1[0],pt1[1]), 5, (0,255,0), cv.FILLED)
        cv.circle(img, (pt2[0],2*pt1[1] - pt2[1]), 5, (0,255,0), cv.FILLED)
        cv.circle(img, (pt3[0],2*pt1[1] - pt3[1]), 5, (0,255,0), cv.FILLED)
        cv.line(img, (pt1[0], pt1[1]), (pt2[0],2*pt1[1] - pt2[1]), (0,255,0), 2)
        cv.line(img, (pt1[0], pt1[1]), (pt3[0],2*pt1[1] - pt3[1]), (0,255,0), 2)


while True :

    if len(pointsList) != 0 and len(pointsList) % 3 == 0:
        getAngle(pointsList)

    cv.imshow("image", img)
    cv.setMouseCallback('image', mousePoints)
    if cv.waitKey(1) & 0xFF == ord('q'):
        pointsList = []
        img = cv.imread(path)
        # img = cv.resize(img, (1000, 968))
        img = cv.resize(img, (1920, 1080))

