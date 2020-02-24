import cv2
import time

cap = cv2.VideoCapture("../asset/720ptest.mp4")
# cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_POS_FRAMES, 28000)
tracker = cv2.TrackerKCF_create()
count = 0
while cap.isOpened():
    t0 = time.time()
    count += 1
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 360))
    if count == 1:
        bbox = cv2.selectROI('Select bbox', frame)
        tracker.init(frame, bbox)
    res, bbox = tracker.update(frame)
    print(time.time()-t0)
    if res:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        cv2.imshow("tracking", frame)
        if cv2.waitKey(33) & 0xff == ord('q'):
            pass
    else:
        cv2.imshow("tracking", frame)
        if cv2.waitKey(33) & 0xff == ord('q'):
            pass


