import time
import cv2


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = ["person"]

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

i=0
while True:
    try:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)

        if len(classIds) != 0:

            last_time = time.time()


            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                img = cv2.rectangle(img, box, (0, 255, 0), thickness=2)
                img = cv2.putText(img, classNames[classId - 1].upper() + str(confs), (box[0], box[1]),
                                  cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

            print(str(i) + "-> Found a Person " + str(confs))
            i+=1
            
        cv2.imshow("Output",img)
        cv2.waitKey(1)



    except Exception as e:
        print(e)
