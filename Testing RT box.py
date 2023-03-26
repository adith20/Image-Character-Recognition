import numpy as np
import cv2
import pickle
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Thoshiba\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract'

width = 300
height = 200
threshold = 0.65
cap = cv2.VideoCapture(0)
cap.set(3, width)  # id 3 is width
cap.set(4, height)  # id 4 is height

pickle_in = open("model_trained.p","rb")
model = pickle.load(pickle_in)


def Preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converting image to grayscale
    image = cv2.equalizeHist(image)  # histogram equalizing for distributing contrast evenly
    image = image/255  # coverting pixel range from [0,255] to [0,1]
    return image



while True:
    success, img = cap.read()  # returns bool value if frame read and the value of the frame


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    himg,wimg,_ = img.shape
    boxes = pytesseract.image_to_boxes(img)
    for b in boxes.splitlines():
        b = b.split(' ')
        x, y, w, h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
        cv2.rectangle(img,(x,himg-y),(w,himg-h),(0,0,255),1)

    cv2.imshow("Original ", img)
    #newimg=img[x:w,himg-h:himg-y]
    #cv2.imshow("detected",newimg)
    #cv2.waitKey(0)
    #cv2.imshow("Original ",img)
    if cv2.waitKey(1) & 0xFF == ord('q'): #allow frame to be shown for a milli sec, quit
        break

