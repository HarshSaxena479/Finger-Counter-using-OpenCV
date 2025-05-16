import os
import cv2
import time
import mediapipe as mp
import hand_tracking_module as htm  # importing the hand tracking module created earlier


wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

pTime = 0

folder_path = "finger count images"
myList = os.listdir(folder_path)
# print(myList)
overlayList = []
for img_path in myList:
    img = cv2.imread(f'{folder_path}/{img_path}')
    overlayList.append(img)
# print(len(overlayList))

detector = htm.handDectector(detectionConfidence=0.75)  # setting the detection confidence as 0.75

'''Tip of each finger:
4 -> tip of thumb
8 -> tip of index finger
12 -> tip of middle finger
16 -> tip of ring finger
20 -> tip of pinky finger
'''
tipids = [4,8,12,16,20]

while cap.isOpened():
    r, frame = cap.read()
    frame = cv2.resize(frame, (900,650))

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)   # landmark list; whenever it detects hand will list down the position of each landmarks
    # print(lmList)

    # if len(lmList) != 0:
    #     if lmList[8][2] < lmList[6][2]: # according to the cv library orientations
    #         print('Index finger open')

    if len(lmList) != 0:
        fingers = []
        #for thumb; if tip is on left it is considered as closed; otherwise open
        if lmList[tipids[0]][1] > lmList[tipids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #for other fingers
        for id in range(1,5):
            if lmList[tipids[id]][2] < lmList[tipids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)
        totalfingers = fingers.count(1)
        print(totalfingers)

        '''overlayList[0] = 1 finger -> f1 img
        overlayList[1] = 2 finger -> f2 img
        overlayList[2] = 3 finger -> f3 img
        overlayList[3] = 4 finger -> f4 img
        overlayList[4] = 5 finger -> f5 img
        overlayList[-1] = 0 finger -> f6 img
        '''
        frame[0:200, 0:200] = overlayList[totalfingers-1]  # putting the image of finger on the window; as windows is also a MATRIX
        cv2.rectangle(frame, (20, 225), (170, 425), (0,255,0), cv2.FILLED)
        cv2.putText(frame, str(totalfingers), (50,350), cv2.FONT_ITALIC, 3, (255,0,0), 3)



    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame,f'FPS : {int(fps)}',(400,70), cv2.FONT_ITALIC,3,(255,0,255),3)

    cv2.imshow('webcam',frame)
    cv2.waitKey(1)