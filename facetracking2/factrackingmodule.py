import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score[0]])
                img = self.fancyDraw(img,bbox)
                
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 255), 2)

        return img, bboxs
    
    def fancyDraw(sel, img, bbox, l=30, t=10, rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        cv2.line(img, (x,y),(x+l,y),(255,0,255),t)
        cv2.line(img, (x,y),(x,y+l),(255,0,255),t)

def main():
    cap = cv2.VideoCapture("D:/pythonprojects/computer-vision/Facetracking/2.mp4")
    pTime = 0
    # Set the desired width for display
    display_width = 600

    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (display_width, int(img.shape[0] * (display_width / img.shape[1]))))
        img, bboxs = detector.findFaces(img)
        
        print(bboxs)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (255, 0, 0), 2)
        # Resize the frame
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
