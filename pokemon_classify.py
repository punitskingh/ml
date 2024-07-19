import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

!wget -O data.zip https://www.dropbox.com/scl/fo/oi9huerppteppcz5t5b32/AJ_ykZC9n5AA0BJat_LlnYI?rlkey=uas4cay1272poo2jc6gf0n5rp&e=1
!unzip data.zip -d "images/"

classes = os.listdir("images/Train/")
print("Training data:")
for f in classes:
    path = "images/Train/" + f
    print(f"{f} - {len(os.listdir(path))}")

print("Test data:")
for f in classes:
    path = "images/Test/" + f
    print(f"{f} - {len(os.listdir(path))}")

train_data = []
train_labels = []

for c in classes:
    folder = f"images/Train/{c}"
    print(f"Processing folder: {folder}")  
    for img_name in os.listdir(folder):
        img_path = f"{folder}/{img_name}"
        img = image.load_img(img_path, target_size=(100, 100))
        img = image.img_to_array(img)
        train_data.append(img)
        train_labels.append(c)

len(train_labels)

train_data = np.array(train_data)
train_labels = np.array(train_labels)

train_data.shape

train_data = train_data.reshape(len(train_data), 30000)
train_data.shape

category2label = {'Pikachu': 0, 'Charmander': 1, 'Bulbasaur': 2}
label2category = {0: 'Pikachu', 1: 'Charmander', 2: 'Bulbasaur'}

train_labels = np.array([category2label[label] for label in train_labels])

train_labels.shape

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)

train_labels.shape

from keras import Sequential
from keras.layers import Dense

n = train_data.shape[1]
n

model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(n,)))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy")

model.summary()

model.fit(train_data, train_labels, batch_size=32, epochs=20)

test_data = []
test_labels = []

for c in classes:
    folder = f"images/Test/{c}"
    print(f"Processing folder: {folder}")
    for img_name in os.listdir(folder):
        img_path = f"{folder}/{img_name}"
        img = image.load_img(img_path, target_size=(100, 100))
        img = image.img_to_array(img)
        test_data.append(img)
        test_labels.append(c)

len(test_data)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

test_data.shape

test_labels = np.array([category2label[label] for label in test_labels])
test_labels = to_categorical(test_labels)

test_labels.shape

test_data = test_data.reshape(len(test_data), 30000)
test_data.shape

model.evaluate(test_data, test_labels)

pred = model.predict(test_data).argmax(axis=1)

t_img = "images/Test/Pikachu/2037.jpg"

img = image.load_img(t_img, target_size=(100, 100))
img = image.img_to_array(img)

sns.set_theme(style='whitegrid')

plt.imshow(img.astype('int'))
plt.show()

img = img.reshape(1, 30000)

img.shape

pred = label2category[model.predict(img).argmax()]
pred
