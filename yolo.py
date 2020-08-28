import cv2
import numpy as np

#load cf and weight

net=cv2.dnn.readNet("E:\yolo\yolov3.weights","E:\yolo\yolov3.cfg.txt")
classes=[]

with open("E:\yolo\model_data_coco_classes.txt","r") as f:
    classes=[line.strip() for line in f.readlines()]

#print(classes)

layers_names=net.getLayerNames()
outputlayers=[layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors=np.random.uniform(0,255,size=(len(classes),3))

#loading the image

img=cv2.imread(r"E:\photos\yolo.jpg")#import image
#img=cv2.resize(img,None,fx=3,fy=3)#enlarge the size

height,width,channel=img.shape


#Detecting the image

blob=cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)
#see what inside the blob
#for b in blob:
    #for n,img_blob in enumerate(b):
        #cv2.imshow(str(n),img_blob)

net.setInput(blob)
outs=net.forward(outputlayers)

#showing information on the screen

boxes=[]
class_ids=[]
confidences=[]
for out in outs:
    for detection in out:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]
        if confidence > 0.5:
            #object detected
            center_x=int(detection[0]*width)
            center_y=int(detection[1]*height)
            w=int(detection[2]*width)
            h=int(detection[3]*height)

            #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
            #Rectangle coordinates
            x=int(center_x-w/2)
            y=int(center_y-h/2)
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#print(len(boxes))
indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
#print(indexes)
for i in indexes.flatten():
    x,y,w,h=boxes[i]
    label=str(classes[class_ids[i]])
    confidence=str(round(confidences[i],2))
    #print(label)
    color=colors[i]
    cv2.rectangle(img, (x, y), (x + w, y + h),color, 2)
    cv2.putText(img,label+" "+confidence,(x+10,y),cv2.FONT_HERSHEY_PLAIN,1,color,1)

cv2.imshow("image",img)#display the image
cv2.waitKey(0)#display the image without haste

#save output image to disk
cv2.imwrite("E:\photos\object-detection.jpg", img)

# release resources
cv2.destroyAllWindows()