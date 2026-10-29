import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self, mode = False, upBody = False, smooth = True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode = self.mode,
                                     model_complexity = 1,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence = self.detectionCon,
                                     min_tracking_confidence = self.trackCon)
        # self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)


    def findPose(self, img, draw = True, altura_desejada=1080):
        # Redimensionando o Video para melhor visualização
        h_original, w_original, _ = img.shape
        fator_escala = altura_desejada / h_original
        nova_largura = int(w_original * fator_escala)
        nova_altura = int(h_original * fator_escala)
        img = cv2.resize(img, (nova_largura, nova_altura))

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h) # Transformando decimal em pixel
                lmList.append([id, cx, cy])
                if(draw):
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList



def main():
    cap = cv2.VideoCapture('PoseVideos/2.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img, altura_desejada=720)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[11])
            cv2.circle(img, (lmList[11][1], lmList[11][2]), 5, (255, 0, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()