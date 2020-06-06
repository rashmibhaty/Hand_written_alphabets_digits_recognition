# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:37:12 2020

@author: rashmibh
"""
import cv2
import os
import matplotlib.pyplot as plt
from keras.models import load_model

MODEL_FILE="model.alphabets.saved"
model = load_model(MODEL_FILE)
image_directory="val_images"
IMAGE_DIM1=28
IMAGE_DIM2=28

#%%
#Using openCV       
list_of_files = os.listdir(image_directory)
for file in list_of_files:
    image_file_name = os.path.join(image_directory, file)
    if ".png" in image_file_name or ".jpeg" in image_file_name:
        print("checking"+ image_file_name)
        
        img = cv2.imread(image_file_name)

        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, img_array) = cv2.threshold(grayImage, 140, 255, cv2.THRESH_BINARY)

        #img_array = cv2.GaussianBlur(img_array, (5, 5), 0)

          
        img_array = cv2.bitwise_not(img_array)

        plt.tight_layout()
        plt.subplot(1, 2, 1)
        plt.imshow(img_array, cmap = plt.cm.binary)
        #plt.show()
        img_size = IMAGE_DIM1
        new_array = cv2.resize(img_array, (img_size,img_size),interpolation=cv2.INTER_AREA )
        
        plt.subplot(1, 2, 2)
        plt.imshow(new_array, cmap = plt.cm.binary)
        
        plt.show()
        
        new_array = new_array.reshape(1, IMAGE_DIM1, IMAGE_DIM2, 1)
        # prepare pixel data
        new_array = new_array.astype('float32')
        new_array = new_array / 255.0
        
        
        predicted = model.predict(new_array)
        pr_val=predicted.argmax()
        print("Predicted value:",pr_val)
        
        if pr_val>=0 and pr_val<=9:
            print("Predicted Digit is:",chr(48+pr_val))
        elif pr_val>=10 and pr_val<=35:
            print("Predicted Capital Alphabet is:",chr(65+pr_val-10))
        elif pr_val>=36 and pr_val<=61:
            print("Predicted Small Alphabet is:",chr(97+pr_val-36))
        print("--------------------------------------------")

