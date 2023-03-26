import numpy as np  # importing numpy package
import cv2  # importing opencv package
import os
from sklearn.model_selection import train_test_split  # importing traintestsplit fn from scikit-learn library
from keras.preprocessing.image import ImageDataGenerator  # for data augmentation
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, AveragePooling2D
import matplotlib.pyplot as plt
import pickle

path = 'data'
myList = os.listdir(path)  # list of all the folders in the data directory
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)  # gives no of folders(A..Z) in the data directory###
imageDimensions = (32, 32, 3)

images = []
classes = []

# Accessing the images from the data directory

for x in range(65, 91):  # access folder names by their ASCII
    myPicList = os.listdir(path + "/" + chr(x))  # contains list of images in the folders in each iteration
    for y in myPicList:
        curImg = cv2.imread(path + "/" + chr(x) + "/" + y)  # loads each image from the folders###
        curImg = cv2.resize(curImg, (32, 32))  # resize the image given the new size(width,height-in pixels)
        images.append(curImg)  # appending each image to the images list
        classes.append(x-64)  # appending each class name(A..Z) for the corresponding image
    print(chr(x), end=" ")
print(" ")

###images list has all the images from all the class folders in the data directory

images = np.array(images)  ###converting images list to numpy array
classes = np.array(classes)
print("No.of images and their size-",
      images.shape)  ###gives no of images and size of the images in the numpyarray(n,height,width,color)



# To split the dataset into Training,Testing

testRatio = 0.2  # testing dataset = 20%
valRatio = 0.2  # validation dataset = 20%
X_train, X_test, y_train, y_test = train_test_split(images, classes,test_size=testRatio)  # splitting the dataset into 80% training & 20% testing
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=valRatio)  # splitting the dataset into 80% training & 20% testing
print("No. of images for training- ", X_train.shape)
print("No. of images for testing- ", X_test.shape)
print("No. of images for validation- ", X_val.shape)

totalsamples = []
for x in range(1,noOfClasses+1):
    print("No.of samples in ", x, "-",len(np.where(y_train == x)[0]))  # getting the no.of samples(after split) in each of the classes
    totalsamples.append(len(np.where(y_train == x)[0]))


# Image Pre-processing

def Preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converting image to grayscale
    image = cv2.equalizeHist(image)  # histogramequalizing for distributing contrast evenly
    image = image / 255  # coverting pixel range from [0,255] to [0,1]
    return image


X_train = np.array(list(map(Preprocessing, X_train)))  # passing training set for preprocessing & converting result to numpyarray
X_test = np.array(list(map(Preprocessing, X_test)))  # passing testing set for preprocessing & converting result to numpyarray
X_val = np.array(list(map(Preprocessing, X_val)))  # passing validation set for preprocessing & converting result to numpyarray

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],1)  # adding a depth of 1(32,32,1) to the dataset for convolution
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,rotation_range=10)  # performing augmentation transformation
dataGen.fit(X_train)  # applying the augmentation to training set

# one hot encoding of matrix        
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
label_encoder3 = LabelEncoder()
vector1 = label_encoder1.fit_transform(y_train)  # converting string(y_train) array into numeric values
y_train = to_categorical(vector1, noOfClasses)  # converting numeric array into binary vector matrix

vector2 = label_encoder2.fit_transform(y_test)
y_test = to_categorical(vector2, noOfClasses)

vector3 = label_encoder3.fit_transform(y_val)
y_val = to_categorical(vector3, noOfClasses)


# creating NN model

def Model():
    noOfFilters = 60  # filters perform multiplication between i/p data and weights and learns feature feature from the image(giving feature map)
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    imageDimensions = (32, 32, 3)
    sizeOfPool = (2, 2)
    nodes = 500

    model = Sequential()  # stacks sequential layers from i/p to o/p
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0], imageDimensions[1], 1),
                      activation='relu')))  # convolutional layer
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(AveragePooling2D(
        pool_size=sizeOfPool))  # Pooling layer -down samples feature maps and max pooling summarizes most active(or predominant feature)in the feature map
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(AveragePooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))  # dropping random neurons during training(50 %)
    model.add(Flatten())  # flatten layer to convert 2D feature matrix to vector for inputting to fully connected layer
    model.add(Dense(nodes, activation='relu'))  # fully connected layer where outputs from conv layers are fed to get predictions
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])  # configuring to train,adam opt algo,categ crossentr-softmax+cross-entropy loss
    return model


model = Model()
print(model.summary())

batchSizeVal = 50
epochsVal = 10
stepsPerEpochVal = 2000

history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batchSizeVal),
                              # training by aug,no.of samples in an iteration,no of steps for 1 epoch,shuffling seq of batches
                              steps_per_epoch=stepsPerEpochVal, epochs=epochsVal, validation_data=(X_val, y_val),
                              shuffle=1)

# Plotting results from the Training
plt.figure(1)  # creating a figure with id 1
plt.plot(history.history[
             'loss'])  # plotting the history of accuracy and losses which are saved in hitory call from training model
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)  # evaluate on testing data
print('Test Score = ', score[0])     #test loss
print('Test Accuracy =', score[1])

# To save the model for testing using webcam
pickle_out = open("Test file/model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
