from django.shortcuts import render

import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tkinter
import numpy as np
import cv2
import imutils
import pyttsx3


# Create your views here.
def index(request):
    return render(request, 'AdminApp/index.html')


def login(request):
    return render(request, 'AdminApp/index.html')


def LogAction(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    if username == 'Admin' and password == 'Admin':
        return render(request, 'AdminApp/AdminHome.html')
    else:
        context = {'data': 'Login Failed ....!!'}
        return render(request, 'AdminApp/index.html', context)


def home(request):
    return render(request, 'AdminApp/AdminHome.html')


global dataset


def loaddataset(request):
    global dataset
    dataset = "dataset"
    context = {'data': 'Traffic Sign Dataset Uploaded Successfully..!!'}
    return render(request, 'AdminApp/AdminHome.html', context)


global training_set, test_set


def ImageGenerate(request):
    global training_set, test_set
    train_datagen = ImageDataGenerator(shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
    test_datagen = ImageDataGenerator()
    training_set = train_datagen.flow_from_directory(dataset,
                                                     target_size=(48, 48),
                                                     batch_size=32,
                                                     save_format='ppm',
                                                     class_mode='categorical',
                                                     shuffle=True)
    test_set = test_datagen.flow_from_directory(dataset,
                                                target_size=(48, 48),
                                                batch_size=32,
                                                save_format='ppm',
                                                class_mode='categorical',
                                                shuffle=False)

    context = {'data': "Generated Training And Testing Images successfully"}
    return render(request, 'AdminApp/AdminHome.html', context)


global classifier


def generateCNN(request):
    global classifier
    if os.path.exists("model\\model_weights.h5"):
        classifier = Sequential()
        classifier.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(48, 48, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(activation="relu", units=128))
        classifier.add(Dense(activation="softmax", units=43))
        classifier.load_weights('model/model_weights.h5')
        context = {"data": "CNN Model Generated Successfully.."}
        return render(request, 'AdminApp/AdminHome.html', context)
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(48, 48, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(activation="relu", units=128))
        classifier.add(Dense(activation="softmax", units=43))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model = classifier.fit_generator(training_set,
                                 steps_per_epoch=150,
                                 epochs=50,
                                 validation_data=test_set,
                                 validation_steps=80)
        classifier.save_weights('model/model_weights.h5')
        final_val_accuracy = model.history['accuracy'][-1]
        msg=f'Final Accuracy: {final_val_accuracy:.4f}'
        context = {"data": "CNN Model Generated Successfully..", "msg":msg}
        return render(request, 'AdminApp/AdminHome.html', context)



def uploadSingImage(request):
    return render(request, 'AdminApp/Upload.html')


global filename, uploaded_file_url


def fileUpload(request):
    global filename, uploaded_file_url
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        location = myfile.name
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        imagedisplay = cv2.imread(BASE_DIR + "/" + uploaded_file_url)
        oringinal = imagedisplay.copy()
        dis_img = imutils.resize(oringinal, width=400)
        cv2.imshow('uploaded Image', dis_img)
        cv2.waitKey(0)
    context = {'data': 'Traffic Sign Uploaded Successfully'}
    return render(request, 'AdminApp/Upload.html', context)




def RecognizeSign(request):

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imagetest = image.load_img(BASE_DIR + "/" + uploaded_file_url, target_size=(48, 48))
    imagetest = image.img_to_array(imagetest)
    imagetest = np.expand_dims(imagetest, axis=0)
    loaded_classifier = Sequential()
    loaded_classifier.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(48, 48, 3), activation='relu'))
    loaded_classifier.add(MaxPooling2D(pool_size=(2, 2)))
    loaded_classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
    loaded_classifier.add(MaxPooling2D(pool_size=(2, 2)))
    loaded_classifier.add(Flatten())
    loaded_classifier.add(Dense(activation="relu", units=128))
    loaded_classifier.add(Dense(activation="softmax", units=43))
    loaded_classifier.load_weights('model/model_weights.h5')
    pred = loaded_classifier.predict(imagetest)

    predict = np.argmax(pred)
    classes = { 0:' Speed limit (20km/h)' ,1:' Speed limit (30km/h)' ,2:' Speed limit (50km/h)' ,3:' Speed limit (60km/h)' ,4:' Speed limit (70km/h)' ,
    5:' Speed limit (80km/h)' ,6:' End of speed limit (80km/h)' ,7:' Speed limit (100km/h)' ,8:' Speed limit (120km/h)' ,9:' No passing' ,
    10:' No passing veh over 3.5 tons' ,11:' Right-of-way at intersection' ,12:' Priority road' ,13:' Yield' ,14:' Stop' ,15:' No vehicles' ,
    16:' Veh > 3.5 tons prohibited' ,17:' No entry' ,18:' General caution' ,19:' Dangerous curve left' ,20:' Dangerous curve right' ,
    21:' Double curve' ,22:' Bumpy road' ,23:' Slippery road' ,24:' Road narrows on the right' ,25:' Road work' ,26:' Traffic signals' ,
    27:' Pedestrians' ,28:' Children crossing' ,29:' Bicycles crossing' ,30:' Beware of ice/snow' ,31:' Wild animals crossing' ,
    32:' End speed + passing limits' ,33:' Turn right ahead' ,34:' Turn left ahead' ,35:' Ahead only' ,36:' Go straight or right' ,
    37:' Go straight or left' ,38:' Keep right' ,39:' Keep left' ,40:' Roundabout mandatory' ,41:' End of no passing' ,42:' End no passing vehicles more than 3.5 tons'  }
    global msg;
    for x in classes.keys():
        if predict == x:
            msg=list(classes.values()) [list(classes.keys()).index(x)]
    data="Traffic Sign Recognized As :"+msg
    imagedisplay = cv2.imread(BASE_DIR + "/" + uploaded_file_url)
    oring = imagedisplay.copy()
    output = imutils.resize(oring, width=400)
    cv2.putText(output, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (1, 204, 0), 2)
    cv2.imshow("Traffic Sign", output)
    engine = pyttsx3.init()
    engine.setProperty("rate", 140)
    engine.setProperty("volume", 1.0)
    engine.say(data)
    engine.runAndWait()
    cv2.waitKey(0)

    context = {'data': data}
    return render(request,'AdminApp/AdminHome.html', context)
