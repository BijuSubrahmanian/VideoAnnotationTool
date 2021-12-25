import cv2

import numpy as np
import time
import sys
from os import system
import os
import string

label=""

#itemlistapplicable=['handbag',
#'suitcase',
#'bottle',
#'wine glass',
#'cup',
#'fork',
#'knife',
#'bowl',
#'banana',
#'remote',
#'keyboard',
#'cell phone',
#'microwave',
#'oven',
#'toaster',
#'sink',
#'book',
#'vase',
#'scissors',
#'hair drier',
#'toothbrush',
#]

#itemlistapplicable=['person','bottle','banana','vase']
itemlistapplicable=['bottle']
#'suitcase'

from darkflow.net.build import TFNet

option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.15
}

tfnet = TFNet(option)

capture = cv2.VideoCapture(0)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
capture.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
i=0

while True:
    stime = time.time()
    ret, frame = capture.read()
    results = tfnet.return_predict(frame)
    if ret:
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
			
            #frame = cv2.rectangle(frame, tl, br, color, 7)
            #frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
            #if label in itemlistapplicable :  #['bottle']: #
            if True:
                print("topleft ",tl,' br '  ,br)
                
                frame = cv2.rectangle(frame, tl, br, color, 7)
                #frame = cv2.putText(frame, label+ "t left " + str(tl[0]) + " t right " +  str(tl[1]) +"\n Bleft "+ str(br[0]) +" BRight " +str(br[1]) ,tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
                #frame = cv2.putText(frame, label+ " - " + str(result['confidence'])+"%" ,tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
                frame = cv2.putText(frame, label+ " - " + str(result['confidence'])+"%" ,tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
                i+=1
                #cv2.imwrite(directory+"/"+ directory +"-" + str(i) + ".jpg", frame)
                        
                #fname=directory+"/"+ directory +"-" + str(i) + ".jpg"
                #fnameabspath=os.path.abspath(fname)
                #fname = str(fname).split('/')[-1:][0]
                
        
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1/(time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
capture.release()
cv2.destroyAllWindows()
