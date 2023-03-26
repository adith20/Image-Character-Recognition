import numpy as np
import cv2
import pickle


width = 500
height = 480
threshold = 0.65
cap = cv2.VideoCapture(0)   # Reading from webcam
cap.set(3, width)  # id 3 is width
cap.set(4, height)  # id 4 is height

pickle_in = open("model_trained.p","rb")
model = pickle.load(pickle_in)
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


def Preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converting image to grayscale
    image = cv2.equalizeHist(image)  # histogram equalizing for distributing contrast evenly
    image = image/255  # coverting pixel range from [0,255] to [0,1]
    return image



while True:
    success, Img = cap.read()  # returns bool value of frame read and the value of the frame - Webcam
    #Img = cv2.imread('s.jpg')     Test Image
    img = np.asarray(Img) # converting to array
    img = cv2.resize(img,(32,32))
    img = Preprocessing(img)
    cv2.imshow("Preprocessed image", img)
    img = img.reshape(1,32,32,1)

    # To predict
    clno = int(model.predict_classes(img))
    classval = classes[clno]
    predictions = (model.predict(img))
    prob = np.amax(predictions)       #max probability value
    print(classval,prob)

    if prob>threshold:
        cv2.putText(Img,str(classval)+" "+str(prob),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

    cv2.imshow("Original ",Img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  #allow frame to be shown for a milli sec, quit
        break

