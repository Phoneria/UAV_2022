import time
import cv2
import math
import haversine as hs
from dronekit import connect


t_file = open("Sentences.txt", "w")
t_file.write(" ")
t_file.close()


flight_time = 10 # minute
photo_limit = 5
wait_for_delete = 3 #second
wait_for_fly = 10 #second

import serial
import RPi.GPIO as GPIO


# Connection
plane = connect("/dev/ttyACM0", wait_ready=False)

while (True):
    print("Waiting For GPS")
    if(plane.location.global_frame.lat!=None):
          break

print("Armable, armed, version, velocity, alt,lat,lon")
print(plane.is_armable)
print(plane.armed)
print(plane.version)
print(plane.velocity)

print(plane.location.global_frame.alt)
print(plane.location.global_frame.lat)
print(plane.location.global_frame.lon)

time.sleep(wait_for_fly)
print("Flying")


def servo_control(repeat, sleep, pvm, freq, x1, x2):

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pvm, GPIO.OUT)
    p = GPIO.PWM(pvm, freq)
    p.start(50)
    i = 0
    try:

        while i < repeat:
            p.ChangeDutyCycle(x1)
            time.sleep(sleep)
            p.ChangeDutyCycle(x2)
            time.sleep(sleep)
            i += 1
    except:
        p.stop()
        GPIO.cleanup()



def drop_ball(person_gps_x, person_gps_y, current_x, current_y, vel):
    
    m, k, A, g = 0.18, 0.5, 0.1, 9.8

    v_lim = math.sqrt(m * g / (k * A))
    ball_drop_time = (30 / v_lim) * (3 / 2)
    
    drop_distance = vel * ball_drop_time
    
    
   
    loc1 = (first_x,first_y)
    loc2 = (current_x,current_y)
    distance = (hs.haversine(loc1, loc2) * 1000)

    if distance <= drop_distance :
        t_file = open("Sentences.txt", "a")
        t_file.write(" V limit: " + str(v_lim))
        t_file.write(" Ball Drop Time: " + str(ball_drop_time))
        t_file.write(" Drop Distance: " + str(drop_distance))
        t_file.write(" Current_x: " + str(current_x))
        t_file.write(" Current_y: " + str(current_y))
        t_file.write(" Current_alt: " + str(plane.location.global_frame.alt))
        t_file.write(" person_gps_x: " + str(person_gps_x))
        t_file.write(" person_gps_y: " + str(person_gps_y))
        t_file.close()
        servo_control(1,1, 33, 80, 5, 12.5)
        servo_control(1, 1, 32, 35, 12.5, 5)
        return True




cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)



classNames=["person"]

configPath ="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightPath ="frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)



i=0
last_time = time.time()
start_time = time.time()
first_x = 0
first_y = 0
photo_founded = False
found_time = time.time()

while (time.time()-start_time<flight_time*60):
    try:
        success,img = cap.read()

        classIds,confs,bbox=net.detect(img,confThreshold = 0.5)


        if time.time()-last_time >= wait_for_delete:
            i = 0

        if len(classIds)!=0:
            last_time = time.time()

            for classId,confidence ,box in zip(classIds.flatten(),confs.flatten(),bbox):
                img = cv2.rectangle(img, box, (0, 255, 0), thickness=2)
                img = cv2.putText(img, classNames[classId - 1].upper()+str(confs), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 2,(0, 255, 0), 2)

            i += 1
            t_file = open("Sentences.txt", "a")
            t_file.write(str(i)+ "-> Person Founded "+str(confs)+"\n")
            t_file.close()
            cv2.imwrite(str(i)+"_"+str(confs)+".png",img)
            print(str(i)+ "-> Person Founded "+str(confs))


            if i>=photo_limit:
                first_x = plane.location.global_frame.lat
                first_y = plane.location.global_frame.lon
                t_file = open("Sentences.txt", "a")
                t_file.write("Person GPS : " + str(first_x) +" "+ str(first_y)+ "\n")
                t_file.close()
                found_time= time.time()
                cv2.imwrite("finally.png",img)
                photo_founded = True



            #cv2.imshow("Output",img)
            #cv2.waitKey(1)
            if photo_founded and time.time()-found_time() >=10:
                drop_ball(first_x,first_y,plane.location.global_frame.lat,plane.location.global_frame.lon,20)
                if drop_ball(first_x,first_y,plane.location.global_frame.lat,plane.location.global_frame.lon,20):
                    t_file = open("Sentences.txt", "a")
                    t_file.write("Ball just left")
                    t_file.close()
                    break
               


    except Exception as e:
        print(e)

t_file.close()
