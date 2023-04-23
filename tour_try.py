import time
import cv2
import math
import haversine as hs
from dronekit import connect

t_file = open("Sentences.txt", "w")

flight_time = 10  # minute
photo_limit = 5
wait_for_delete = 3  # second
wait_for_fly = 10  # second
drop_dist_initial = 10
quit_range_wait_time = 15  # second
raund_meter =20

import serial
import RPi.GPIO as GPIO

# Connection
# plane = connect("/dev/ttyACM0", wait_ready=False)# usb
plane = connect("/dev/serial0", wait_ready=False)  # telemetry

tour = 0
while (True):
    print(str(tour) + "-) Waiting For GPS")

    time.sleep(0.1)
    tour += 1
    if (plane.location.global_frame.lat != None):
        break

print("Armable : ", plane.is_armable)
print("Armed : ", plane.armed)
print("Version : ", plane.version)
print("Velocity : ", plane.velocity)

print("Alt : ", plane.location.global_frame.alt)
print("Lat : ", plane.location.global_frame.lat)
print("Lon : ", plane.location.global_frame.lon)
print("Waiting for " + str(wait_for_fly) + " seconds")

time.sleep(wait_for_fly)
t_file.write("Flying..." + "\n")
print("Flying...")


def servo_control(repeat, sleep, pvm, freq, x1, x2):
    print("Servo is working ....")
    t_file = open("Sentences.txt", "a")
    t_file.write("Servo is working on: " + str(pvm) + "\n")
    t_file.close()
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
    print("Dropping Ball...")

    m, k, A, g = 0.18, 0.5, 0.1, 9.8

    v_lim = math.sqrt(m * g / (k * A))
    ball_drop_time = (30 / v_lim) * (3 / 2)

    drop_distance = vel * ball_drop_time

    loc1 = (first_x, first_y)
    loc2 = (current_x, current_y)
    distance = (hs.haversine(loc1, loc2) * 1000)

    print("Distance = "+ str(distance)+ " Drop_Distance = "+ str(drop_distance) + " Initialized Drop Distance = "+ str(drop_dist_initial))
    if distance <= drop_dist_initial:
        print("Ball has just dropped")
        t_file = open("Sentences.txt", "a")
        t_file.write(" ")
        t_file.write(" V limit: " + str(v_lim) + "\n")
        t_file.write(" Ball Drop Time: " + str(ball_drop_time) + "\n")
        t_file.write(" Drop Distance: " + str(drop_distance) + "\n")
        t_file.write(" Current_x: " + str(current_x) + "\n")
        t_file.write(" Current_y: " + str(current_y) + "\n")
        t_file.write(" Current_alt: " + str(plane.location.global_frame.alt) + "\n")
        t_file.write(" person_gps_x: " + str(person_gps_x) + "\n")
        t_file.write(" person_gps_y: " + str(person_gps_y) + "\n")
        t_file.close()
        servo_control(1, 1, 33, 80, 5, 12.5)
        servo_control(1, 1, 32, 35, 12.5, 5)

start_x = plane.location.global_frame.lat
start_y = plane.location.global_frame.lon
print("Start x : " + str(start_x))
print("Start y : " + str(start_y))

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

i = 0
last_time = time.time()
start_time = time.time()
first_x = 0
first_y = 0
photo_founded = False
found_time = time.time()
t_file.write("Searching For Person" + "\n")
print("Searching For Person...")
t_file.close()

serial_photo_timer = time.time()
serial_holder = 1

tour_timer =time.time()
raund = 0
diff_between_raunds = 0
raund_time = 0
second_raund_started =False
while (time.time() - start_time < flight_time * 60):
    try:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        try:

            start_location = (start_x, start_y)
            current_location = ( plane.location.global_frame.lat,plane.location.global_frame.lon)
            difference = (hs.haversine(start_location, current_location) * 1000)

            if (difference <=raund_meter and time.time()-tour_timer >= 10):
                raund+=1
                if raund==2 and not second_raund_started :
                    diff_between_raunds = time.time()-tour_timer
                    print("Difference Between Raunds " + str(diff_between_raunds))
                    raund_time = time.time()
                    second_raund_started = True
                tour_timer = time.time()

                print("Raund : "+ str(raund))

            if raund ==3:
                if (time.time()-raund_time >= diff_between_raunds):
                    print("Raund = 3, dropping balls")
                    servo_control(1, 1, 33, 80, 5, 12.5)
                    servo_control(1, 1, 32, 35, 12.5, 5)
        except Exception as e:
            print("Exception : " + e)

        if time.time() - serial_photo_timer >= 0.2:
            cv2.imwrite("Serial_photo_" + str(serial_holder) + ".png", img)
            serial_photo_timer = time.time()
            serial_holder += 1

        if time.time() - last_time >= wait_for_delete:
            i = 0
            print("i = 0")
        if len(classIds) != 0:
            last_time = time.time()

            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                img = cv2.rectangle(img, box, (0, 255, 0), thickness=2)
                img = cv2.putText(img, classNames[classId - 1].upper() + str(confs), (box[0], box[1]),
                                  cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

            i += 1
            t_file = open("Sentences.txt", "a")
            t_file.write(str(i) + "-> Found a Person " + str(confs) + "\n")
            t_file.close()
            cv2.imwrite(str(i) + "_" + str(confs) + ".png", img)
            print(str(i) + "-> Found a Person " + str(confs))

            if i >= photo_limit and not photo_founded:
                first_x = plane.location.global_frame.lat
                first_y = plane.location.global_frame.lon
                t_file = open("Sentences.txt", "a")
                t_file.write("Person GPS : " + str(first_x) + " " + str(first_y) + "\n")
                t_file.close()
                found_time = time.time()
                cv2.imwrite("finally.png", img)
                photo_founded = True
                print("Found a Person, GPS : " + str(first_x) + " " + str(first_y) + "\n")


            # cv2.imshow("Output",img)
            # cv2.waitKey(1)
        if photo_founded and time.time() - found_time() >= quit_range_wait_time:
            print("Out of quit range, searching for cycle")
            drop_ball(first_x, first_y, plane.location.global_frame.lat, plane.location.global_frame.lon, 20)





    except Exception as e:
        pass
