#@Author : Biju Subrahmanian
#AnnotateWithAugmentation
import Augmentor
import cv2
import os
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
        'model': 'cfg/yolo.cfg',
        'load': 'bin/yolov2.weights',
        'threshold': 0.15
    }

tfnet = TFNet(option)


def annotateAugedFiles(filepath,itemtype,labelname,frameheight=1080,framewidth=1920) :
    #print('itemtype',itemtype)
    itemlistapplicable=[itemtype]
    #itemlistapplicable=['car','bus','truck']
    #'suitcase'

    image_folder = filepath
    dirFiles=os.listdir(image_folder)
    dirFiles.sort()
    sorted(dirFiles)
    AnnotationTemplate="""<annotation>
        <folder>$product</folder>
        <filename>$filename</filename>
        <path>$filenamewithpath</path>
        <source>
        <database>Unknown</database>
        </source>
        <size>
        <width>$wdt</width>
        <height>$hgt</height>
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
        #ProductLabel=sys.argv[1]
        ProductLabel=labelname
        print('label ',ProductLabel)
    except:
        #system('say Enter label name for the product !')
        print('label not found ',ProductLabel)
        sys.exit()

    ### Creating directory for label annotation
    directory =ProductLabel
    if not os.path.exists(directory):
        os.makedirs(directory)
    images = [img for img in dirFiles if img.endswith(".jpg")]
    
    #capture = cv2.VideoCapture(filename)
    #capture = cv2.VideoCapture('UnorderedInstrument.mp4')
    colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT,1920)

    #frame_cnt=0
    #i=0
    #Flag=True
    for image in images:
        stime = time.time()
        frame = cv2.imread(os.path.join(image_folder, image))
        frame1=frame
        #frame_cnt+=1
        ret=True
        #if frame_cnt % 3 == False:
        #    continue
        try:
            results = tfnet.return_predict(frame)
        except:
            print("ERROR")
            Flag=False
            continue

        if ret:
            for color, result in zip(colors, results):
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                print('labeldetected',label)
                #frame = cv2.rectangle(frame, tl, br, color, 7)
                #frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
                if label in itemlistapplicable :  #['bottle']: #
                    print("topleft ",tl,' br '  ,br)
                    frame1=frame.copy()
                    #frame = cv2.rectangle(frame, tl, br, color, 7)
                    #frame = cv2.putText(frame, label+ "t left " + str(tl[0]) + " t right " +  str(tl[1]) +"\n Bleft "+ str(br[0]) +" BRight " +str(br[1]) ,tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
                    #frame = cv2.putText(frame, label+ " - " + str(result['confidence'])+"%" ,tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
                    
                    #frame = cv2.putText(frame, ProductLabel+ " - " + str(result['confidence'])+"%" ,tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
                    #i+=1
                    fnamexml=os.path.splitext(os.path.basename(image))[0] + '.xml'
                    
                    #cv2.imwrite(directory+"/"+ directory +"-" + str(i) + ".jpg", frame1)
                            
                    #fname=directory+"/"+ directory +"-" + str(i) + ".jpg"
                    #fnamexml=directory+"/"+ directory +"-" + str(i) + ".xml"
                    fnameabspath=os.path.abspath(image_folder + "/" + image)
                    fname = image#str(fname).split('/')[-1:][0]
                    print(fname)
                    print(fnameabspath)
                    AnnotationTemplateCurrent=AnnotationTemplate
                    AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$tl',str(tl[0]))
                    AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$tr',str(tl[1]))
                    AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$br',str(br[1]))
                    AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$bl',str(br[0]))
                    AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$product',image_folder)

                    AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$filenamewithpath',fnameabspath)
                    AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$filename',fname)
                    AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$wdt',str(framewidth))
                    AnnotationTemplateCurrent=AnnotationTemplateCurrent.replace('$hgt',str(frameheight))
                    print(fnamexml)
                    f = open(image_folder+ "/" + fnamexml, "w")
                    f.write(AnnotationTemplateCurrent)
            
            print('FPS {:.1f}'.format(1/(time.time() - stime)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    
    cv2.destroyAllWindows()

#processVideo(filename='surgicalscissorstraight.mp4',itemtype='scissors',labelname='surgicalscstraight5inch',frameheight=1080,framewidth=1920)

def doAug (filepath,flipyesno='no',zoomyesno='no',sample=100):
    p = Augmentor.Pipeline(filepath)
    p.rotate(probability=1, max_left_rotation=3, max_right_rotation=3)
    if flipyesno =='yes':
        p.flip_left_right(probability=0.5)
    if zoomyesno =='yes':
        p.zoom_random(probability=0.5, percentage_area=0.999)
#p.flip_top_bottom(probability=0.5)
    p.sample(sample)
    # Start annotating

#doAug (filepath='C:/biju/Experiments/yoloexp/Real-Time-Object-Detection/darkflow/tupbottle',flipyesno='no',zoomyesno='no',sample=100)
#annotateAugedFiles(filepath=filepath+ '/output',itemtype='bottle',labelname='tupbottle',frameheight=1080,framewidth=1920)    

#annotateAugedFiles(filepath='C:/biju/Experiments/yoloexp/Real-Time-Object-Detection/darkflow/tupbottle/output',itemtype='bottle',labelname='tupbottle',frameheight=1080,framewidth=1920)