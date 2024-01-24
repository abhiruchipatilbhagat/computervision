import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose 
pose = mpPose.Pose()

cap = cv2.VideoCapture('D:/pythonprojects/computer-vision/posturetracking/2.mp4')

# Set the desired width for display
display_width = 600
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)


    # Resize the frame
    img = cv2.resize(img, (display_width, int(img.shape[0] * (display_width / img.shape[1]))))

    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


