import cv2

import numpy as np
import time
import sys
from os import system
import os
import string
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os.path



label=""

itemlistapplicable=['car']

AnnotationTemplate="""<annotation>
    <folder>$product</folder>
    <filename>$filename</filename>
    <path>$filenamewithpath</path>
    <source>
    <database>Unknown</database>
    </source>
    <size>
    <width>1920</width>
    <height>1080</height>
    <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
    <name>$product</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
    <xmin>$tl</xmin>
    <ymin>$tr</ymin>
    <xmax>$bl</xmax>
    <ymax>$br</ymax>
    </bndbox>
    </object>
    </annotation>
"""

try:
    label=sys.argv[1]
    print('label ',label)
except:
    system('say Enter label name for the product !')
    print('label not found ',label)
    sys.exit()

### Creating directory for label annotation
directory =label
if not os.path.exists(directory):
    os.makedirs(directory)

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
            if label in itemlistapplicable :  #['bottle']: #
                print("topleft ",tl,' br '  ,br)
                
                frame = cv2.rectangle(frame, tl, br, color, 7)
                #frame = cv2.putText(frame, label+ "t left " + str(tl[0]) + " t right " +  str(tl[1]) +"\n Bleft "+ str(br[0]) +" BRight " +str(br[1]) ,tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
                crop_img=frame[int(tl[1]):int(br[1]),int(tl[0]):int(br[0])]
                height, width, channels=crop_img.shape
                frame = cv2.putText(frame, label+ " - " + str(result['confidence'])+"%" ,tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
                
                i+=1
                #crop_img=frame[int(yminn):int(ymaxx),int(xminn):int(xmaxx)]
                
                
                #Creating common width
                
                scaled_width=200
                scaled_factor=scaled_width/width
                scaled_height=scaled_factor *height
                
                scaled_img=cv2.resize(crop_img, (scaled_width, int(scaled_height)))
                color_histogram_feature_extraction.color_histogram_of_test_image(scaled_img)

                prediction = knn_classifier.main('training.data', 'test.data')
                print ('Color of the object : ',prediction)
                cv2.imwrite(directory+"/"+ directory +"-" + str(i) + ".jpg", scaled_img)
                        
                fname=directory+"/"+ directory +"-" + str(i) + ".jpg"
                fnamexml=directory+"/"+ directory +"-" + str(i) + ".xml"
                fnameabspath=os.path.abspath(fname)
                fname = str(fname).split('/')[-1:][0]
                print(fname)
                print(fnameabspath)
                AnnotationTemplateCurrent=AnnotationTemplate
                AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$tl',str(tl[0]))
                AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$tr',str(tl[1]))
                AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$br',str(br[1]))
                AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$bl',str(br[0]))
                AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$product',directory)

                AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$filenamewithpath',fnameabspath)
                AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$filename',fname)
                f = open(fnamexml, "w")
                f.write(AnnotationTemplateCurrent)
        #print(directory+"/"+ directory +"-" + str(i) + ".jpg")
        
        
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1/(time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
capture.release()
cv2.destroyAllWindows()
