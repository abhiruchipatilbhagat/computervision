import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, staticMode, model_complexity, smooth, enable_segmentation, smooth_segmentation, minDetectionCon, minTrackCon):

            self.staticMode = staticMode
            self.modelComplexity = model_complexity
            self.smooth = smooth
            self.enableSegmentation = enable_segmentation
            self.smoothSegmentation = smooth_segmentation
            self.minDetectionCon = minDetectionCon
            self.minTrackCon = minTrackCon

            self.mpDraw = mp.solutions.drawing_utils
            self.mpPose = mp.solutions.pose
            self.pose = self.mpPose.Pose(static_image_mode = self.staticMode,model_complexity = self.modelComplexity, smooth_landmarks = self.smooth, enable_segmentation = self.enableSegmentation, smooth_segmentation = self.smoothSegmentation, min_detection_confidence = self.minDetectionCon, min_tracking_confidence = self.minTrackCon)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
        
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:  
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList


    

    


def main():
    cap = cv2.VideoCapture('D:/pythonprojects/computer-vision/posturetracking/2.mp4')

    # Set the desired width for display
    display_width = 800
    p_time = 0

    # Provide values for the parameters when creating an instance
    detector = poseDetector(
        staticMode=True,  # Replace with the actual value
        model_complexity=1,  # Replace with the actual value
        smooth=True,  # Replace with the actual value
        enable_segmentation=True,  # Replace with the actual value
        smooth_segmentation=True,  # Replace with the actual value
        minDetectionCon=0.5,  # Replace with the actual value
        minTrackCon=0.5  # Replace with the actual value
    )

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        print(lmList)
        

        # Resize the frame
        img = cv2.resize(img, (display_width, int(img.shape[0] * (display_width / img.shape[1]))))
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
