import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askopenfilename

img_size = 128

img_to_predict = askopenfilename(initialdir = "E:/CurrentProject/Final_Project_Sem7/test")



def prepare_image(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = img/255
    img = cv2.resize(img, (img_size, img_size))
    return img.reshape(-1, img_size, img_size, 1)

print("Loading trained model...")
model = tf.keras.models.load_model("Concrete_Crack_Classification_model.model")
print("Trained model loaded!")

print("Model predicting...")

prediction = model.predict([prepare_image(img_to_predict)])

if (prediction[0][0] > (0.5) ):
    pred_text = "Networks prediction:\nThis surface DOES NOT have a crack on it. ".format((1 - prediction[0][0]) * 100)

elif    (prediction[0][0] <= 0.5):
        pred_text = "Networks prediction:\nThis surface DOES have a crack on it. ".format((1 - prediction[0][0]) * 100)
else:
    print("\nSomething went wrong...")
    
plt.imshow(cv2.resize(cv2.imread(img_to_predict), (img_size, img_size)))
plt.title("What the Neural Network is receiving as input:")
plt.text(2, 145, pred_text, fontweight = "bold", color = "red")
#plt.text(2,10,prediction[0][0],fontweight = "bold", color = "red")
plt.show()
