import socket
import cv2
import time
from PIL import Image

def compress(image_file):
    image = Image.open(image_file)
    image.save("frame_2.jpeg", "jpeg", optimize = True, quality = 10)
    return

server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(("", 9999)) #8888 9999 5297
server.listen(5)
cap = cv2.VideoCapture(0)
cap.set(3,320) # width is id no 3
cap.set(4,240) # height is id no 4
cap.set(10,100) # Brightness id is 10

clientsocket, address = server.accept()

while True:
    success, img = cap.read()
    cv2.imwrite("frame.jpeg", img)
    compress("frame.jpeg")
    with open("frame_2.jpeg","rb") as image:
        read_image = image.read()
        byteimg = bytearray(read_image)
        print("CONVERT TO BYTE")
    clientsocket.send(byteimg)
    print("SEND") 
    time.sleep(0.5)
