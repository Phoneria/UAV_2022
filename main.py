import time
import cv2
import math
import haversine as hs
from dronekit import connect

t_file = open("Sentences.txt", "w")

flight_time = 3 # minute
photo_limit = 2
wait_for_delete = 3  # second
wait_for_fly = 10  # second
drop_dist_initial = 10 # meter
quit_range_wait_time = 15  # second
start_circle =  20 #meter

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

start_x = plane.location.global_frame.lat
start_y = plane.location.global_frame.lon

print(str(start_x) +" "+ str(start_y))

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


def get_dist(x1,y1,x2,y2):
    location_1 = (x1, y1)
    location_2 = (x2, y2)
    return (hs.haversine(location_1, location_2) * 1000)


def drop_ball(current_x, current_y, vel):
    print("Dropping Ball...")

    m, k, A, g = 0.18, 0.5, 0.1, 9.8

    v_lim = math.sqrt(m * g / (k * A))
    ball_drop_time = (30 / v_lim) * (3 / 2)

    drop_distance = vel * ball_drop_time

    loc1 = (first_x, first_y)
    loc2 = (current_x, current_y)
    distance = (hs.haversine(loc1, loc2) * 1000)

    print("Distance = " + str(distance) + " Drop_Distance = " + str(
        drop_distance) + " Initialized Drop Distance = " + str(drop_dist_initial))

    t_file = open("Sentences.txt", "a")
    t_file.write("Distance = " + str(distance) + " Drop_Distance = " + str(
        drop_distance) + " Initialized Drop Distance = " + str(drop_dist_initial))
    t_file.close()
    if distance <= drop_dist_initial:
        print("Ball has just dropped")
        t_file = open("Sentences.txt", "a")
        t_file.write("Ball has just dropped")

        t_file.close()
        for i in range(5):
            servo_control(1, 1, 33, 80, 5, 12.5)

        return True



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
is_it_over = False


raunt = 0
bool_raunt =True
tour_timer = time.time() # Start time of each raunt
second_tour = 0
each_tour = 0

while (time.time() - start_time < flight_time * 60):
    try:
        try:
            if (bool_raunt and get_dist(start_x,start_y,plane.location.global_frame.lat,plane.location.global_frame.lon)<=start_circle):
                raunt+=1
                tour_timer = time.time()
                print(str(raunt))
                if raunt==2:
                    second_tour = time.time()

                elif raunt == 3:
                    each_tour = time.time()-second_tour


                elif raunt == 4:
                    current_time = time.time()
                    while time.time()-current_time<=(9*each_tour/20):
                        if (time.time()-current_time>=2*60):
                            break
                        print("Going through the target")

                    for i in range(5):
                        servo_control(1, 1, 33, 80, 5, 12.5)
                    print("Dropped Ball in 4th Tour")
                    break

                t_file = open("Sentences.txt", "a")
                t_file.write(str(raunt))
                t_file.close()
                bool_raunt = False

            if (not bool_raunt and get_dist(start_x,start_y,plane.location.global_frame.lat,plane.location.global_frame.lon)>=start_circle):
                bool_raunt = True
        except Exception as e:
            print(e)

        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)

        if time.time() - serial_photo_timer >= 0.2:
            cv2.imwrite("Serial_photo_" + str(serial_holder) + ".png", img)
            serial_photo_timer = time.time()
            serial_holder += 1
            print(str(serial_holder))
        if time.time() - last_time >= wait_for_delete:
            i = 0
            print("i = 0")
        if not photo_founded and len(classIds) != 0:

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



        if photo_founded and (time.time() - found_time) >= quit_range_wait_time:
            print("Out of quit range, searching for cycle")
            while True:
                if(drop_ball(plane.location.global_frame.lat, plane.location.global_frame.lon, 20)):
                    is_it_over = True
                    break

        if is_it_over:
            break

    except Exception as e:
        print(e)



"""
Rahmân ve Rahîm olan Allah'ın adıyla.

Necm39-42: Şüphesiz insana kendi emeğinden başkası yoktur.

Şüphesiz onun çalışması ileride görülecektir.

Sonra çalışmasının karşılığı kendisine tastamam verilecektir.

Şüphesiz en son varış Rabbinedir


Bakara 255: Allah'tan başka hiçbir İlah yoktur. O, daima yaşayan, daima duran,

bütün varlıkları ayakta tutandır. O'nu ne gaflet basar, ne de uyku.

Göklerdeki ve yerdeki herşey O'nundur. O'nun izni olmadan huzurunda şefaat etmek kimin haddine!

Onların önlerinde ve arkalarında ne varsa hepsini bilir.

Onlar ise, O'nun dilediği kadarından başka ilminden hiçbir şey kavrayamazlar.

O'nun hükümdarlığı, bütün gökleri ve yeri kucaklamıştır. Her ikisini görüp gözetmek,

ona bir ağırlık da vermez. O, çok yüce, çok büyüktür.


"""









