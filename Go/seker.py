import cv2
import time
import math
import haversine as hs
from dronekit import connect
import socket
import time

# from PIL import Image
try:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("", 5297))
    server.listen(5)
    clientsocket, address = server.accept()
except Exception as e:
    print(e)

t_file = open("Sentences.txt", "w")

flight_time = 10  # minute
photo_limit = 5
wait_for_delete = 3  # second
wait_for_fly = 10  # second
drop_dist_initial = 30
quit_range_wait_time = 15  # second

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


# print("Flying...")


def servo_control(repeat, sleep, pvm, freq, x1, x2):
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
    m, k, A, g = 0.18, 0.5, 0.1, 9.8

    v_lim = math.sqrt(m * g / (k * A))
    ball_drop_time = (30 / v_lim) * (3 / 2)

    drop_distance = vel * ball_drop_time

    loc1 = (first_x, first_y)
    loc2 = (current_x, current_y)
    distance = (hs.haversine(loc1, loc2) * 1000)

    if distance <= drop_dist_initial:
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
# print("Searching For Person...")
t_file.close()

serial_photo_timer = time.time()
serial_holder = 1

while (time.time() - start_time < flight_time * 60):
    try:
        success, img = cap.read()
        try:
            cv2.imwrite("frame.jpeg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open("frame.jpeg", "rb") as image:
                read_image = image.read()
                byteimg = bytearray(read_image)
            clientsocket.send(byteimg)
        except Exception as e:
            print(e)

        classIds, confs, bbox = net.detect(img, confThreshold=0.5)

        if time.time() - serial_photo_timer >= 0.2:
            cv2.imwrite("Serial_photo_" + str(serial_holder) + ".png", img)
            serial_photo_timer = time.time()
            serial_holder += 1

        if time.time() - last_time >= wait_for_delete:
            i = 0

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
            # print(str(i) + "-> Found a Person " + str(confs))

            if i >= photo_limit and not photo_founded:
                first_x = plane.location.global_frame.lat
                first_y = plane.location.global_frame.lon
                t_file = open("Sentences.txt", "a")
                t_file.write("Person GPS : " + str(first_x) + " " + str(first_y) + "\n")
                t_file.close()
                found_time = time.time()
                cv2.imwrite("finally.png", img)
                photo_founded = True

            # cv2.imshow("Output",img)
            # cv2.waitKey(1)
        if photo_founded and time.time() - found_time() >= quit_range_wait_time:
            drop_ball(first_x, first_y, plane.location.global_frame.lat, plane.location.global_frame.lon, 20)





    except Exception as e:
        pass
