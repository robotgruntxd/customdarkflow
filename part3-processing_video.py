import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import serial


arduinoPort = serial.Serial("COM5", 9600, timeout=1)
time.sleep(2)


def mover(n):

    # arduinoPort.write(b'%s' % (n,))
    arduinoPort.write(bytes(n, encoding='utf-8'))


option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.15,
    'gpu': 1.0
}



tfnet = TFNet(option)

capture = cv2.VideoCapture(0)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    results = tfnet.return_predict(frame)
    if ret:
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)


            height, width = frame.shape[:2]
            if result['label'] == "persona" and result['confidence'] > 0.75:
                diff = tl[0] - (width - br[0])
                if diff > 5:
                    mover('d')
                elif diff < -5:
                    mover('i')
            cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        arduinoPort.close()
        break


