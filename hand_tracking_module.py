import cv2
import mediapipe as mp
import time


class handDectector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = float(detectionConfidence)
        self.trackConfidence = float(trackConfidence)

        self.mpHands = mp.solutions.hands
        # self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        #                                 self.detectionConfidence, self.trackConfidence)
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConfidence,  # Ensure correct parameters
            min_tracking_confidence=self.trackConfidence
        )
        self.mpDraw = mp.solutions.drawing_utils  # this utility allow us to show several points on our hand without us doing the mathematical calculations

    def findHands(self, frame, draw=True):
        # convert our image to RGB
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # process the frame for us
        # print(results.multi_hand_landmarks) #show something if detects something

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # it will draw the hand for us

        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        lmList = []  # landmark list which contains the positions of each landmark if detected
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]  #for a particular hand not for all


            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # position of center
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList



def main():
    pTime = 0  # previous time for FPS
    cTime = 0  # current time for FPS
    cap = cv2.VideoCapture(0)
    detector = handDectector()

    while True:
        r, frame = cap.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        if len(lmList)!=0:
            print(lmList[20])   # this will tell us the position of our small finger of our hand

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # showing the FPS
        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('videocam', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
