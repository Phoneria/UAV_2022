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
                if raunt==1:
                    second_tour = time.time()
    
                elif raunt == 2:
                    each_tour = time.time()-second_tour
    
    
                elif raunt == 5:
                    current_time = time.time()
                    while time.time()-current_time<=(9*each_tour/20):
                        print("Going through the target")
    
                    for i in range(5):
                        servo_control(1, 1, 33, 80, 5, 12.5)
                    print("Dropped Ball in 5th Tour")
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

for i in range(5):
    servo_control(1, 1, 33, 80, 5, 12.5)


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

Fetih Suresi :
. Şüphesiz biz sana apaçık bir fetih verdik.(1)

(1) Âyetteki "fetih" ile daha sonra gerçekleşecek Mekke fethi kastedilmektedir. Ayrıca sûrenin inmesinden önce gerçekleşen ve Mekke fethine zemin hazırlamış olan Hudeybiye barışının kastedilmiş olması da mümkündür.
2,3. Ta ki Allah, senin geçmiş ve gelecek günahlarını bağışlasın, sana olan nimetini tamamlasın, seni doğru yola iletsin ve Allah sana, şanlı bir zaferle yardım etsin.

4. O, inananların imanlarını kat kat artırmaları için kalplerine huzur ve güven indirendir. Göklerin ve yerin orduları Allah'ındır. Allah, hakkıyla bilendir, hüküm ve hikmet sahibidir.

5. Bütün bunlar Allah'ın; inanan erkek ve kadınları, içlerinden ırmaklar akan, içinde temelli kalacakları cennetlere koyması, onların kötülüklerini örtmesi içindir. İşte bu, Allah katında büyük bir başarıdır.

6. Bir de, Allah'ın, hakkında kötü zanda bulunan münafık erkeklere ve münafık kadınlara, Allah'a ortak koşan erkeklere ve Allah'a ortak koşan kadınlara azap etmesi içindir. Kötülük girdabı onların başına olsun! Allah onlara gazap etmiş, onları lânetlemiş ve kendilerine cehennemi hazırlamıştır. Orası ne kötü bir varış yeridir!

7. Göklerin ve yerin orduları Allah'ındır. Allah, mutlak güç sahibidir, hüküm ve hikmet sahibidir.

8. (Ey Muhammed!) Şüphesiz biz seni bir şâhit, bir müjdeci ve bir uyarıcı olarak gönderdik.

9. Ey insanlar! Allah'a ve Peygamberine inanasınız, ona yardım edesiniz, ona saygı gösteresiniz ve sabah akşam Allah'ı tespih edesiniz diye (Peygamber'i gönderdik.)

10. Sana bîat edenler ancak Allah'a bîat etmiş olurlar.(2) Allah'ın eli onların ellerinin üzerindedir. Verdiği sözden dönen kendi aleyhine dönmüş olur. Allah'a verdiği sözü yerine getirene, Allah büyük bir mükâfat verecektir.

(2) "Bîat", el tutuşup söz vermek demektir. Âyette, Hudeybiye'de müslümanların, Hz. Peygamber'e bağlılık göstereceklerine, gerektiğinde onunla birlikte savaşacaklarına dair söz vermeleri kastedilmektedir. Bu olay, İslâm tarihinde "Bey'atu'r-Rıdvan" diye anılır.
11. Bedevîlerin (savaştan) geri bırakılanları sana, "Bizi mallarımız ve ailelerimiz alıkoydu; Allah'tan bizim için af dile" diyecekler. Onlar kalplerinde olmayanı dilleriyle söylerler. De ki: "Allah, sizin bir zarara uğramanızı dilerse, yahut bir yarar elde etmenizi dilerse, O'na karşı kimin bir şeye gücü yeter? Hayır, Allah, yaptıklarınızdan haberdardır."

12. (Ey münafıklar!) Siz aslında, Peygamberin ve inananların bir daha ailelerine geri dönmeyeceklerini sanmıştınız. Bu, sizin gönüllerinize güzel gösterildi de kötü zanda bulundunuz ve helâki hak eden bir kavim oldunuz.

13. Kim Allah'a ve Peygambere inanmazsa bilsin ki, şüphesiz biz, inkârcılar için alevli bir ateş hazırladık.

14. Göklerin ve yerin hükümranlığı Allah'ındır. O, dilediğini bağışlar, dilediğine ceza verir. Allah, çok bağışlayandır, çok merhamet edendir.

15. Savaştan geri bırakılanlar, siz ganimetleri almaya giderken, "Bırakın biz de sizinle gelelim" diyeceklerdir. Onlar Allah'ın sözünü değiştirmek isterler. De ki: "Siz bizimle asla gelmeyeceksiniz. Allah, önceden böyle buyurmuştur." Onlar, "Bizi kıskanıyorsunuz" diyeceklerdir. Hayır, onlar pek az anlarlar.

16. Bedevîlerin (savaştan) geri bırakılanlarına de ki: "Siz, güçlü kuvvetli bir kavme karşı teslim oluncaya kadar savaşmaya çağrılacaksınız. Eğer itaat ederseniz, Allah size güzel bir mükâfat verir. Ama önceden döndüğünüz gibi yine dönerseniz, Allah sizi elem dolu bir azaba uğratır."

17. Köre güçlük yoktur, topala güçlük yoktur, hastaya güçlük yoktur. (Bunlar savaşa katılmak zorunda değillerdir.) Kim Allah'a ve Peygamberine itaat ederse, Allah onu, içlerinden ırmaklar akan cennetlere koyar. Kim de yüz çevirirse, onu elem dolu bir azaba uğratır.

18,19. Şüphesiz Allah, ağaç altında sana bîat ederlerken inananlardan hoşnut olmuştur. Gönüllerinde olanı bilmiş, onlara huzur, güven duygusu vermiş ve onlara yakın bir fetih(3) ve elde edecekleri birçok ganimetler nasip etmiştir. Allah mutlak güç sahibidir, hüküm ve hikmet sahibidir.

(3) Âyette sözü edilen fetih, Hudeybiye barışından hemen sonra gerçekleşen Hayber'in fethi olayıdır. Daha sonraki âyetlerde sözü edilen ganimetler de burada elde edilen ganimetlerdir.
20. Allah, size, elde edeceğiniz birçok ganimetler vaad etmiştir. Şimdilik bunu size hemen vermiş ve insanların ellerini sizden çekmiştir. (Allah, böyle yaptı) ki, bunlar mü'minler için bir delil olsun, sizi de doğru bir yola iletsin.

21. Henüz elde edemediğiniz, fakat Allah'ın, ilmiyle kuşattığı başka (kazançlar) da vardır. Allah, her şeye hakkıyla gücü yetendir.

22. İnkâr edenler sizinle savaşsalardı, arkalarını dönüp kaçarlar, sonra da ne bir dost, ne de bir yardımcı bulabilirlerdi.

23. Allah'ın öteden beri işleyip duran kanunu (budur). Allah'ın kanununda asla bir değişiklik bulamazsın.

24. O, Mekke'nin göbeğinde, sizi onlara karşı üstün kıldıktan sonra, onların ellerini sizden, sizin ellerinizi onlardan çekendir. Allah, yaptıklarınızı hakkıyla görmektedir.

25. Onlar, inkâr edenler ve sizi Mescid-i Haram'ı ziyaretten ve (ibadet amacıyla) bekletilen kurbanlıkları yerlerine ulaşmaktan alıkoyanlardır. Eğer, oradaki henüz tanımadığınız inanmış erkeklerle, inanmış kadınları bilmeyerek ezmeniz ve böylece size bir eziyet gelecek olmasaydı, (Allah, Mekke'ye girmenize izin verirdi). Allah, dilediğini rahmetine koymak için böyle yapmıştır. Eğer, inananlarla inkârcılar birbirinden ayrılmış olsalardı, onlardan inkâr edenleri elem dolu bir azaba uğratırdık.

26. Hani inkâr edenler kalplerine taassubu, cahiliye taassubunu yerleştirmişlerdi. Allah ise, Peygamberine ve inananlara huzur ve güvenini indirmiş ve onların takva (Allah'a karşı gelmekten sakınma) sözünü tutmalarını sağlamıştı. Zaten onlar buna lâyık ve ehil idiler. Allah, her şeyi hakkıyla bilmektedir.

27. Andolsun, Allah, Peygamberinin rüyasını doğru çıkardı. Allah dilerse, siz güven içinde başlarınızı kazıtmış veya saçlarınızı kısaltmış olarak, korkmadan Mescid-i Haram'a gireceksiniz. Allah, sizin bilmediğinizi bildi ve size bundan başka yakın bir fetih daha verdi.(4)

(4) Âyette sözü edilen "yakın fetih" Mekke fethinden önce gerçekleşen Hayber fethi veya Hudeybiye barışıdır. Hudeybiye barışının fetih diye nitelenmesi, İslâm adına önemli açılımlar sağlamış olması sebebiyledir.
28. O, Peygamberini hidayet ve hak din ile gönderendir. (Allah) o hak dini bütün dinlere üstün kılmak için (böyle yaptı). Şahit olarak Allah yeter.

29. Muhammed, Allah'ın Resûlüdür. Onunla beraber olanlar, inkârcılara karşı çetin, birbirlerine karşı da merhametlidirler. Onların, rükû ve secde hâlinde, Allah'tan lütuf ve hoşnutluk istediklerini görürsün. Onların secde eseri olan alametleri yüzlerindedir. İşte bu, onların Tevrat'ta ve İncil'de anlatılan durumlarıdır: Onlar filizini çıkarmış, onu kuvvetlendirmiş, kalınlaşmış, gövdesi üzerine dikilmiş, ziraatçıların hoşuna giden bir ekin gibidirler. Allah, kendileri sebebiyle inkârcıları öfkelendirmek için onları böyle sağlam ve dirençli kılar. Allah, içlerinden iman edip salih amel işleyenlere bir bağışlama ve büyük bir mükâfat vaad etmiştir.


"""









