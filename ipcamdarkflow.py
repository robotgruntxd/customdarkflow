import datetime

import cv2
from darkflow.net.build import TFNet
import numpy as np
import time



option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.5,  # 0 = more boxes, 1 = less boxes
    'gpu': 1.0
}

tfnet = TFNet(option)

colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

url = "rtsp://192.168.1.10:554/user=admin&password=&channel=1&stream=0.sdp?real_stream--rtp-caching=100"
camera = cv2.VideoCapture(url)

while camera.isOpened():
    # print("[INFO] Camera connected at " + datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"))
    # attempts = reset_attempts()
    # recall = process_video(attempts)
    # stime = time.time()
    ret, frame = camera.read()
    # try:
    results = None

    if frame is None:
        camera.release()
        camera = cv2.VideoCapture(url)


    if frame is not None:
        results = tfnet.return_predict(frame)
    # except:
    #     print('error given')
    #     run = 0
    #     break
    if results is not None:
        if ret:
            for color, result in zip(colors, results):
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                frame = cv2.rectangle(frame, tl, br, color, 7)
                frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                height, width = frame.shape[:2]
            cv2.imshow('frame', frame)
            # print('FPS {:.1f}'.format(1 / (time.time() - stime)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

